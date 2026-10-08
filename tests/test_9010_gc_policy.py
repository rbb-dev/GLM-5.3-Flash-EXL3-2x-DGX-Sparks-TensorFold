"""Patch 9010 (when the engine's process collects garbage), on the CPU with a fake collector: a full collection walks
every list, and the engine keeps prompt states' ids as lists of up to a million ints (one full collection held the GIL
153 ms with 16 of them on an x86 core, 10 ms with arrays), so automatic collection is off, the engine's loop collects
the young generations as Python would, and a full collection runs only when the engine is idle (or after a long
busy spell)."""

from tensorfold.families.glm5_next.cuda.gc_policy import GcPolicy


class FakeGc:
    def __init__(self):
        self.calls, self.count = [], [0, 0, 0]

    def freeze(self):
        self.calls.append("freeze")

    def disable(self):
        self.calls.append("disable")

    def get_count(self):
        return tuple(self.count)

    def collect(self, generation=2):
        self.calls.append(f"collect({generation})")
        if generation >= 1:
            self.count = [0, 0, self.count[2]]
        return 0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_start_freezes_what_startup_made_and_turns_automatic_collection_off():
    g = FakeGc()
    GcPolicy(gc=g, clock=Clock()).start()
    assert g.calls == ["freeze", "disable"]


def test_a_busy_loop_collects_only_the_young_generations_when_python_would():
    g, clock = FakeGc(), Clock()
    p = GcPolicy(gc=g, clock=clock, young=700)
    p.start()
    g.calls.clear()
    g.count[0] = 699
    p.tick(idle=False)
    assert g.calls == []
    g.count[0] = 700
    p.tick(idle=False)
    assert g.calls == ["collect(1)"]


def test_a_full_collection_runs_when_idle_and_not_again_at_once():
    g, clock = FakeGc(), Clock()
    p = GcPolicy(gc=g, clock=clock, idle_gap_s=30.0)
    p.start()
    g.calls.clear()
    clock.now += 31.0
    p.tick(idle=True)
    p.tick(idle=True)
    assert g.calls == ["collect(2)"]
    clock.now += 31.0
    p.tick(idle=True)
    assert g.calls == ["collect(2)", "collect(2)"]


def test_a_long_busy_spell_still_gets_a_full_collection_now_and_then():
    g, clock = FakeGc(), Clock()
    p = GcPolicy(gc=g, clock=clock, force_s=900.0)
    p.start()
    g.calls.clear()
    clock.now += 899.0
    p.tick(idle=False)
    assert g.calls == []
    clock.now += 2.0
    p.tick(idle=False)
    assert g.calls == ["collect(2)"]


def test_the_policy_can_be_turned_off():
    g = FakeGc()
    p = GcPolicy(gc=g, clock=Clock(), enabled=False)
    p.start()
    g.count[0] = 10_000
    p.tick(idle=True)
    assert g.calls == []


def test_the_scheduler_ticks_before_it_waits_for_work_and_on_every_busy_iteration():
    from tensorfold.families.glm5_next.cuda.multi import GlmScheduler

    ticks = []

    class Policy:
        def tick(self, idle):
            ticks.append(idle)

    class Decoder:
        gc_policy, watchdog = Policy(), 0
        live_streams = 1

        def live(self):
            return self.live_streams

        def round(self):
            return []

        def finish(self, done):
            pass

        def drop(self):
            return []

    class Waiting:
        def get(self):
            return None

    sched = object.__new__(GlmScheduler)
    sched.decoder, sched.held, sched.loading, sched.ready, sched.gather_s = Decoder(), None, [], [], 0.0
    sched.waiting = Waiting()
    sched._cancel_waiting = sched._yield = sched._requeue = lambda: None
    sched._reply = lambda *a: None
    sched._admit = lambda first, until: []
    sched._iteration()                          # busy: streams decode
    sched.decoder.live_streams = 0
    sched._iteration()                          # idle: the scheduler waits for a request
    assert ticks == [False, True]
