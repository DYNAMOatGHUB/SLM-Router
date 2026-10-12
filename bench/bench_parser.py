"""NFR-8 parser microbenchmark (PRD-03 A3-7).

Runs every parser truth-table input 1,000 times and checks p99 < 1 ms.

    python -m bench.bench_parser [--repeat 1000] [--limit-ms 1.0]
"""

from __future__ import annotations

import argparse
import statistics
import time

from tests.test_parser import TRUTH_TABLE
from tinyagent.parser import parse


def run(repeat: int) -> list[float]:
    timings_ms: list[float] = []
    for _ in range(repeat):
        for _, text, _ in TRUTH_TABLE:
            started = time.perf_counter_ns()
            parse(text)
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return timings_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--limit-ms", type=float, default=1.0)
    args = parser.parse_args(argv)

    timings = sorted(run(args.repeat))
    p99 = timings[min(len(timings) - 1, round(0.99 * (len(timings) - 1)))]
    print(f"samples: {len(timings)}")
    print(f"median:  {statistics.median(timings):.4f} ms")
    print(f"p99:     {p99:.4f} ms (limit {args.limit_ms} ms)")
    print(f"max:     {timings[-1]:.4f} ms")
    passed = p99 < args.limit_ms
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
