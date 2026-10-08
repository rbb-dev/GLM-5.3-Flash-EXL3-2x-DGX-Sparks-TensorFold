"""Patch 9008 (the engine's stall meter), on the CPU with a fake clock: while streams decode, how long they wait
between two decode rounds, and what the scheduler did in between (admit, prompt chunks, finish, replies), so a stall
is measured inside the engine, not behind the stream smoothing a client sees."""

from tensorfold.families.glm5_next.cuda.stall_meter import StallMeter


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_a_gap_between_decode_rounds_is_measured_with_what_filled_it():
    clock = Clock()
    m = StallMeter(clock=clock, wall=lambda: 1_700_000_000.0)
    m.round_start()
    clock.now += 0.02
    m.round_end(still_decoding=True)
    with m.phase("admit"):
        clock.now += 0.30
    with m.phase("round"):
        clock.now += 0.20
    clock.now += 0.10
    m.round_start()
    h = m.health()
    assert h["worst_s"] == 0.6 and h["gaps"] == 1
    (event,) = h["recent"]
    assert event["gap_s"] == 0.6 and event["parts"] == {"admit": 0.3, "round": 0.2, "other": 0.1}


def test_no_gap_is_counted_across_idle_time():
    clock = Clock()
    m = StallMeter(clock=clock)
    m.round_start()
    m.round_end(still_decoding=False)          # the last stream finished: nothing waits
    clock.now += 600.0
    m.round_start()
    assert m.health()["gaps"] == 0 and m.health()["worst_s"] == 0.0


def test_only_the_part_of_a_phase_after_the_last_round_counts():
    clock = Clock()
    m = StallMeter(clock=clock)
    with m.phase("round"):                     # a scheduler step that ran a decode round inside it
        clock.now += 0.05
        m.round_start()
        clock.now += 0.02
        m.round_end(still_decoding=True)
        clock.now += 0.15                      # its own work after the round: part of the next gap
    clock.now += 0.05
    m.round_start()
    (event,) = m.health()["recent"]
    assert event["gap_s"] == 0.2 and event["parts"] == {"round": 0.15, "other": 0.05}


def test_small_gaps_are_counted_but_only_long_ones_are_listed():
    clock = Clock()
    m = StallMeter(clock=clock, listed_s=0.05, keep=3)
    m.round_start()
    for gap in (0.01, 0.07, 0.3, 0.6, 1.2, 0.02):
        m.round_end(still_decoding=True)
        clock.now += gap
        m.round_start()
    h = m.health()
    assert h["gaps"] == 6 and h["worst_s"] == 1.2
    assert h["over"] == {"0.05": 4, "0.1": 3, "0.25": 3, "0.5": 2, "1": 1}
    assert [e["gap_s"] for e in h["recent"]] == [0.3, 0.6, 1.2]          # the last three listed, oldest first


def test_a_listed_gap_carries_the_wall_time_it_ended():
    clock = Clock()
    m = StallMeter(clock=clock, wall=lambda: 1_700_000_123.5)
    m.round_start()
    m.round_end(still_decoding=True)
    clock.now += 0.4
    m.round_start()
    assert m.health()["recent"][0]["ended"] == 1_700_000_123.5


def test_the_scheduler_names_a_slow_admission_in_the_gap_it_made():
    from tensorfold.families.glm5_next.cuda.multi import GlmScheduler

    clock = Clock()
    meter = StallMeter(clock=clock)

    class Decoder:
        stalls, watchdog = meter, 0

        def live(self):
            return 1

        def round(self):                        # one decode round: streams decode from here on
            meter.round_start()
            clock.now += 0.02
            meter.round_end(still_decoding=True)
            return []

        def finish(self, done):
            pass

        def drop(self):
            return []

    def slow_admit(first, until):
        clock.now += 0.4
        return []

    sched = object.__new__(GlmScheduler)        # its loop thread is not needed: one iteration at a time
    sched.decoder, sched.held, sched.loading, sched.ready, sched.gather_s = Decoder(), None, [], [], 0.0
    sched._cancel_waiting = sched._yield = sched._requeue = lambda: None
    sched._reply = lambda *a: None
    sched._admit = slow_admit
    sched._iteration()
    sched._iteration()
    (event,) = meter.health()["recent"]
    assert event["gap_s"] == 0.4 and event["parts"] == {"admit": 0.4}


def test_health_says_the_engines_own_time_so_a_reader_on_another_machine_can_place_its_gaps():
    m = StallMeter(clock=Clock(), wall=lambda: 1_700_000_123.4567)
    assert m.health()["now"] == 1_700_000_123.457
