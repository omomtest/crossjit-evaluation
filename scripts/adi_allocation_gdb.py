"""GDB Python: observe NumPy array-data allocation calls and successful returns.

The array-data breakpoints are disabled during import and warmup. The existing
worker's Provider snapshots delimit the window, without new calls in the kernel
or its caller. Python tracing and NumPy memory handlers stay intact.
"""
import json
import os

import gdb

gdb.execute("set breakpoint pending on")
gdb.execute("set pagination off")
gdb.execute("set print thread-events off")
gdb.execute("set disable-randomization off")

records = []
active = False
windows = 0
snapshot_calls = 0
pending_returns = 0
errors = []


def frames():
    result = []
    frame = gdb.newest_frame()
    while frame is not None and len(result) < 16:
        result.append(frame.name())
        frame = frame.older()
    return result


def register(name):
    return int(gdb.parse_and_eval("$" + name))


class AllocationReturn(gdb.FinishBreakpoint):
    def __init__(self, event):
        global pending_returns
        super().__init__(gdb.newest_frame(), internal=True)
        self.silent = True
        self.event = event
        pending_returns += 1

    def stop(self):
        global pending_returns
        self.event["pointer"] = register("rax")
        self.event["successful"] = self.event["pointer"] != 0
        pending_returns -= 1
        return False

    def out_of_scope(self):
        global pending_returns
        errors.append("allocation return left scope")
        pending_returns -= 1


class Allocation(gdb.Breakpoint):
    def __init__(self, name, zeroed=False):
        super().__init__(name, internal=True)
        self.silent = True
        self.enabled = False
        self.zeroed = zeroed
        self.symbol = name

    def stop(self):
        if not active:
            errors.append("allocation breakpoint active outside the window")
            return False
        # x86-64 System V ABI: size, or nmemb and size for zeroed allocations.
        size = register("rdi") * register("rsi") if self.zeroed else register("rdi")
        event = dict(event="array_data_allocation", ordinal=len(records),
                     symbol=self.symbol, requested_bytes=size, frames=frames())
        records.append(event)
        AllocationReturn(event)
        return False


allocations = [Allocation("PyDataMem_UserNEW"),
               Allocation("PyDataMem_UserNEW_ZEROED", zeroed=True)]


def begin():
    global active, windows
    if active:
        errors.append("nested observation window")
    active = True
    windows += 1
    for bp in allocations:
        bp.enabled = True


def end():
    global active
    if not active or pending_returns:
        errors.append("invalid window end or unresolved allocation returns")
    for bp in allocations:
        bp.enabled = False
    active = False


class StartAfterSnapshot(gdb.FinishBreakpoint):
    def __init__(self):
        super().__init__(gdb.newest_frame(), internal=True)
        self.silent = True

    def stop(self):
        begin()
        return False


class Snapshot(gdb.Breakpoint):
    def stop(self):
        global snapshot_calls
        snapshot_calls += 1
        warmup = int(os.environ["CJ_ALLOCATION_WARMUP"])
        repeat = int(os.environ["CJ_ALLOCATION_REPEAT"])
        # initial snapshot; one per invocation; an extra preparation snapshot
        # at warmup. Start after the last warmup invocation's regular snapshot.
        if snapshot_calls == warmup + 2:
            StartAfterSnapshot()
        elif snapshot_calls == warmup + repeat + 2:
            end()
        return False


def exited(event):
    expected_calls = int(os.environ["CJ_ALLOCATION_WARMUP"]) + int(os.environ["CJ_ALLOCATION_REPEAT"]) + 3
    if snapshot_calls != expected_calls:
        errors.append("unexpected snapshot sequence")
    if active:
        errors.append("window left open")
    with open(os.environ["CJ_ALLOCATION_EVENTS"], "w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with open(os.environ["CJ_ALLOCATION_OBSERVER"], "w") as stream:
        json.dump(dict(windows=windows, events=len(records), errors=errors,
                       snapshot_calls=snapshot_calls, expected_snapshot_calls=expected_calls,
                       pending_returns=pending_returns, inferior_exit_code=event.exit_code,
                       complete=windows == 1 and not errors and not pending_returns and event.exit_code == 0), stream, indent=2)


Snapshot("numpy_crossjit_get_stats", internal=True).silent = True
gdb.events.exited.connect(exited)
