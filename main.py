#!/usr/bin/env python3
"""
Standalone Dredd mutation-testing runner for zstd, driven by datagen + a roundtrip
oracle. No dependency on the dredd_test_runners package: the few helpers it needs
(a timeout-aware process runner and the Dredd mutation-tree parser) are inlined
below, adapted from mc-imperial/dredd-compiler-testing.

Dredd mutates the zstd *library* built into the roundtrip binary, so mutants are
selected at runtime via DREDD_ENABLED_MUTATION and nothing is recompiled per test.

Required, in order:
  * mutation_info_file              - Dredd JSON for the mutated build.
  * tracking_info_file              - Dredd JSON for the mutant-tracking build.
                                      The two trees MUST match (non-negotiable sanity check).
  * mutated_zstd                    - checkout root; must contain `datagen` and `roundtrip`.
  * mutant_tracking_zstd            - checkout root; must contain `roundtrip`.

Per iteration:
  1. Pick a seed; claim it by creating work/tests/<seed>/ (atomic; if it already
     exists, another worker has it -> try a different seed).
  2. datagen -s<seed>  ->  work/tests/<seed>/<seed>.dat
  3. Baseline: <mutated_zstd>/roundtrip <dat> <level> with no mutation; must exit 0.
  4. Coverage: <mutant_tracking_zstd>/roundtrip <dat> <level> with
     DREDD_MUTANT_TRACKING_FILE set; read the reached mutant ids.
  5. For each reached, not-yet-killed mutant: <mutated_zstd>/roundtrip <dat> <level>
     with DREDD_ENABLED_MUTATION=<id>. Killed on timeout, crash (signal), or a
     different exit code than the baseline (the roundtrip oracle / a library error).

Embarrassingly parallel: run as many copies as you like against the same work/
directory. The per-seed dir is the atomic claim; work/killed_mutants/<id> is the
atomic kill record.
"""
import argparse
import datetime
import functools
import json
import os
import random
import signal
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import AnyStr, Dict, List, Optional, Set


# --------------------------------------------------------------------------------------
# Timeouts. A 64 KB level-19 roundtrip is ~30 ms on a clean build (process spawn
# included); the Dredd-instrumented builds are somewhat slower, and the per-mutant
# timeout scales off the measured baseline. Upstream dredd-compiler-testing uses
# floor=1.0s, multiplier=5.0, runtime ceiling=10s -- sized for compiling whole
# programs and far too loose here, where a roundtrip exceeding ~1s means a hang we
# want to catch fast, not wait on.
# --------------------------------------------------------------------------------------
RUNTIME_TIMEOUT: int = 5             # one-off ceiling for datagen / baseline / tracking (was 10)
MUTANT_TIMEOUT_FLOOR: float = 0.25   # minimum per-mutant timeout, seconds          (was 1.0)
MUTANT_TIMEOUT_MULTIPLIER: float = 3.0  # per-mutant timeout = max(floor, this*baseline) (was 5.0)


# --------------------------------------------------------------------------------------
# Inlined process runner (adapted from dredd-compiler-testing common/).
# --------------------------------------------------------------------------------------
class ProcessResult:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_process_with_timeout(cmd: List[str],
                             timeout_seconds: float,
                             env: Optional[Dict[AnyStr, AnyStr]] = None,
                             cwd: Optional[Path] = None) -> Optional[ProcessResult]:
    """Run cmd; return ProcessResult, or None if it exceeded timeout_seconds.

    Uses a fresh process group so a timed-out process (and any children it spawned)
    can be torn down together.
    """
    process = None
    try:
        process = subprocess.Popen(cmd, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env=env, cwd=cwd)
        out, err = process.communicate(timeout=timeout_seconds)
        return ProcessResult(returncode=process.returncode, stdout=out, stderr=err)
    except subprocess.TimeoutExpired:
        if process is not None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return None


# --------------------------------------------------------------------------------------
# Inlined Dredd mutation-tree parser (adapted from dredd-compiler-testing common/),
# used only for the mandatory cross-check that the two builds describe the same mutants.
# --------------------------------------------------------------------------------------
def _mutation_ids_for_group(group) -> List[int]:
    for key in ("replaceExpr", "replaceBinaryOperator", "replaceUnaryOperator"):
        if key in group:
            return [inst["mutationId"] for inst in group[key]["instances"]]
    assert "removeStmt" in group
    return [group["removeStmt"]["mutationId"]]


def _mutation_ids_for_node(node) -> List[int]:
    assert "mutationGroups" in node
    return functools.reduce(lambda x, y: x + y,
                            map(_mutation_ids_for_group, node["mutationGroups"]), [])


class MutationTreeNode:
    def __init__(self, mutation_ids, children):
        self.mutation_ids = mutation_ids
        self.children = children


class MutationTree:
    def __init__(self, json_data):
        self.nodes = {}
        self.parent_map = {}
        self.mutation_id_to_node_id = {}
        self.num_mutations = 0
        self.num_nodes = 0

        def populate(json_node, node_id):
            children = []
            for _child in json_node["children"]:
                child_id = self.num_nodes
                children.append(child_id)
                self.parent_map[child_id] = node_id
                self.num_nodes += 1
            self.nodes[node_id] = MutationTreeNode(_mutation_ids_for_node(json_node), children)
            self.num_mutations = max(self.num_mutations,
                                     functools.reduce(max, self.nodes[node_id].mutation_ids, 0))
            for mutation_id in self.nodes[node_id].mutation_ids:
                self.mutation_id_to_node_id[mutation_id] = node_id

        for file_info in json_data["infoForFiles"]:
            for tree_node in file_info["mutationTree"]:
                root_id = self.num_nodes
                self.num_nodes += 1
                populate(tree_node, root_id)


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
class ZstdKillStatus(Enum):
    SURVIVED = 1
    KILL_RUNTIME_TIMEOUT = 2       # roundtrip hung past the timeout
    KILL_CRASH = 3                 # roundtrip died by signal (SIGSEGV/SIGABRT/...)
    KILL_DIFFERENT_EXIT_CODES = 4  # roundtrip oracle failed / library error (nonzero exit)


def still_testing(start_time_for_overall_testing: float,
                  time_of_last_kill: float,
                  total_test_time: int,
                  maximum_time_since_last_kill: int) -> bool:
    if 0 < total_test_time < int(time.time() - start_time_for_overall_testing):
        return False
    if 0 < maximum_time_since_last_kill < int(time.time() - time_of_last_kill):
        return False
    return True


def run_roundtrip_with_mutant(mutant: int,
                              roundtrip_executable: Path,
                              dat_file: Path,
                              level: int,
                              baseline_result: ProcessResult,
                              baseline_run_time: float) -> ZstdKillStatus:
    """Enable exactly one mutant and re-run the roundtrip on the same data file."""
    env = os.environ.copy()
    env["DREDD_ENABLED_MUTATION"] = str(mutant)
    env.pop("DREDD_MUTANT_TRACKING_FILE", None)

    timeout_seconds = max(MUTANT_TIMEOUT_FLOOR, MUTANT_TIMEOUT_MULTIPLIER * baseline_run_time)
    result = run_process_with_timeout(
        cmd=[str(roundtrip_executable), str(dat_file), str(level)],
        timeout_seconds=timeout_seconds, env=env)

    if result is None:
        return ZstdKillStatus.KILL_RUNTIME_TIMEOUT
    if result.returncode < 0:
        return ZstdKillStatus.KILL_CRASH
    if result.returncode != baseline_result.returncode:
        return ZstdKillStatus.KILL_DIFFERENT_EXIT_CODES
    return ZstdKillStatus.SURVIVED


def main():
    start_time_for_overall_testing = time.time()
    time_of_last_kill = start_time_for_overall_testing

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mutation_info_file", type=Path,
                        help="Dredd JSON for the mutated build (required).")
    parser.add_argument("tracking_info_file", type=Path,
                        help="Dredd JSON for the mutant-tracking build (required); "
                             "must describe the same mutants as mutation_info_file.")
    parser.add_argument("mutated_zstd", type=Path,
                        help="Root dir of the Dredd-mutated zstd checkout "
                             "(must contain `datagen` and `roundtrip`).")
    parser.add_argument("mutant_tracking_zstd", type=Path,
                        help="Root dir of the mutant-tracking zstd checkout "
                             "(must contain `roundtrip`).")
    parser.add_argument("--level", default=19, type=int,
                        help="Compression level passed to roundtrip (default: 19).")
    parser.add_argument("--datagen-size", type=int,
                        help="If set, passed to datagen as -g<bytes> (kept fixed across the run).")
    parser.add_argument("--compressibility", type=int,
                        help="If set, passed to datagen as -P<percent> 0-100 (kept fixed).")
    parser.add_argument("--seed", type=int,
                        help="Seed for this runner's own RNG (choice of datagen seeds).")
    parser.add_argument("--run-timeout", default=RUNTIME_TIMEOUT, type=int,
                        help=f"Ceiling (s) for datagen / baseline / tracking runs "
                             f"(default: {RUNTIME_TIMEOUT}).")
    parser.add_argument("--total-test-time", default=86400, type=int,
                        help="Total testing budget in seconds; 0 = unlimited (default: 24h).")
    parser.add_argument("--maximum-time-since-last-kill", default=86400, type=int,
                        help="Stop if no kill within this many seconds; 0 = unlimited (default: 24h).")
    parser.add_argument("--max-invocations", default=0, type=int,
                        help="Stop after this many seeds have been tried; 0 = unlimited.")
    args = parser.parse_args()

    datagen_exe = (args.mutated_zstd / "datagen").resolve()
    mutated_roundtrip = (args.mutated_zstd / "roundtrip").resolve()
    tracking_roundtrip = (args.mutant_tracking_zstd / "roundtrip").resolve()
    for exe in (datagen_exe, mutated_roundtrip, tracking_roundtrip):
        if not exe.is_file():
            parser.error(f"expected executable not found: {exe}")

    # Mandatory sanity check: the two builds must describe the same mutation tree.
    print("Checking that the two mutation trees match...")
    with open(args.mutation_info_file) as f:
        mutation_tree = MutationTree(json.load(f))
    with open(args.tracking_info_file) as f:
        tracking_tree = MutationTree(json.load(f))
    assert mutation_tree.mutation_id_to_node_id == tracking_tree.mutation_id_to_node_id
    assert mutation_tree.parent_map == tracking_tree.parent_map
    assert mutation_tree.num_nodes == tracking_tree.num_nodes
    assert mutation_tree.num_mutations == tracking_tree.num_mutations
    print(f"Check complete: {mutation_tree.num_mutations} mutations across the two builds.")

    if args.seed is not None:
        random.seed(args.seed)

    killed_mutants: Set[int] = set()
    unkilled_mutants: Set[int] = set(range(0, mutation_tree.num_mutations))

    Path("work").mkdir(exist_ok=True)
    Path("work/tests").mkdir(exist_ok=True)
    Path("work/killed_mutants").mkdir(exist_ok=True)

    invocations = 0
    while still_testing(total_test_time=args.total_test_time,
                        maximum_time_since_last_kill=args.maximum_time_since_last_kill,
                        start_time_for_overall_testing=start_time_for_overall_testing,
                        time_of_last_kill=time_of_last_kill):
        if args.max_invocations and invocations >= args.max_invocations:
            break
        invocations += 1

        seed = random.randint(0, 2 ** 31 - 1)
        test_dir = Path("work/tests", str(seed))
        try:
            test_dir.mkdir()
        except FileExistsError:
            print(f"Seed {seed} already claimed; choosing another.")
            continue
        dat_file = test_dir / f"{seed}.dat"

        # datagen -s<seed> [-g<size>] [-P<pct>] -> <seed>.dat
        datagen_cmd = [str(datagen_exe), f"-s{seed}"]
        if args.datagen_size is not None:
            datagen_cmd.append(f"-g{args.datagen_size}")
        if args.compressibility is not None:
            datagen_cmd.append(f"-P{args.compressibility}")
        datagen_result = run_process_with_timeout(cmd=datagen_cmd, timeout_seconds=args.run_timeout)
        if datagen_result is None or datagen_result.returncode != 0 or not datagen_result.stdout:
            print(f"datagen failed for seed {seed}; skipping.")
            continue
        with open(dat_file, "wb") as f:
            f.write(datagen_result.stdout)

        # Baseline: mutated roundtrip with no mutation; must pass.
        baseline_env = os.environ.copy()
        baseline_env.pop("DREDD_ENABLED_MUTATION", None)
        baseline_env.pop("DREDD_MUTANT_TRACKING_FILE", None)
        run_time_start = time.time()
        baseline_result = run_process_with_timeout(
            cmd=[str(mutated_roundtrip), str(dat_file), str(args.level)],
            timeout_seconds=args.run_timeout, env=baseline_env)
        baseline_run_time = time.time() - run_time_start
        if baseline_result is None:
            print(f"Baseline timed out (seed {seed}); skipping.")
            continue
        if baseline_result.returncode != 0:
            print(f"Baseline failed unmutated (seed {seed}, rc={baseline_result.returncode}); skipping.")
            continue

        # Coverage: tracking roundtrip, same data + level.
        tracking_file = test_dir / "covered_mutants.txt"
        if tracking_file.exists():
            os.remove(tracking_file)
        tracking_env = os.environ.copy()
        tracking_env.pop("DREDD_ENABLED_MUTATION", None)
        tracking_env["DREDD_MUTANT_TRACKING_FILE"] = str(tracking_file)
        tracking_result = run_process_with_timeout(
            cmd=[str(tracking_roundtrip), str(dat_file), str(args.level)],
            timeout_seconds=args.run_timeout, env=tracking_env)
        if tracking_result is None or not tracking_file.exists():
            print(f"Tracking run produced no coverage (seed {seed}); skipping.")
            continue

        analysis_timestamp_start = datetime.datetime.now()
        with open(tracking_file) as f:
            covered_by_this_test = sorted({int(line.strip()) for line in f if line.strip() != ""})

        candidate_mutants = [m for m in covered_by_this_test if m not in killed_mutants]
        print(f"[{invocations}] seed {seed}: {len(covered_by_this_test)} mutants reached, "
              f"{len(candidate_mutants)} to try (baseline {baseline_run_time:.3f}s).")
        already_killed_by_other_tests = [m for m in covered_by_this_test if m in killed_mutants]
        killed_by_this_test: List[int] = []
        survived_by_this_test: List[int] = []

        for mutant in candidate_mutants:
            if not still_testing(total_test_time=args.total_test_time,
                                 maximum_time_since_last_kill=args.maximum_time_since_last_kill,
                                 start_time_for_overall_testing=start_time_for_overall_testing,
                                 time_of_last_kill=time_of_last_kill):
                break

            mutant_path = Path("work/killed_mutants", str(mutant))
            if mutant_path.exists():
                killed_mutants.add(mutant)
                unkilled_mutants.discard(mutant)
                already_killed_by_other_tests.append(mutant)
                continue

            status = run_roundtrip_with_mutant(
                mutant=mutant, roundtrip_executable=mutated_roundtrip, dat_file=dat_file,
                level=args.level, baseline_result=baseline_result,
                baseline_run_time=baseline_run_time)

            if status == ZstdKillStatus.SURVIVED:
                survived_by_this_test.append(mutant)
                continue

            killed_mutants.add(mutant)
            unkilled_mutants.discard(mutant)
            killed_by_this_test.append(mutant)
            time_of_last_kill = time.time()
            print(f"Kill! mutant {mutant} via {status}. Mutants killed so far: {len(killed_mutants)}")
            try:
                mutant_path.mkdir()
                with open(mutant_path / "kill_info.json", "w") as outfile:
                    json.dump({"killing_seed": seed,
                               "dat_file": str(dat_file),
                               "level": args.level,
                               "kill_type": str(status),
                               "reproduce": (f"DREDD_ENABLED_MUTATION={mutant} "
                                             f"{mutated_roundtrip} {dat_file} {args.level}"),
                               "kill_timestamp": str(datetime.datetime.now())},
                              outfile, indent=2)
            except FileExistsError:
                print(f"Mutant {mutant} was independently killed by another worker.")

        terminating = not still_testing(
            total_test_time=args.total_test_time,
            maximum_time_since_last_kill=args.maximum_time_since_last_kill,
            start_time_for_overall_testing=start_time_for_overall_testing,
            time_of_last_kill=time_of_last_kill)
        all_considered = sorted(killed_by_this_test + survived_by_this_test
                                + already_killed_by_other_tests)
        terminated_early = (covered_by_this_test != all_considered)
        if terminated_early:
            assert terminating

        with open(test_dir / "kill_summary.json", "w") as outfile:
            json.dump({"terminated_early": terminated_early,
                       "seed": seed,
                       "level": args.level,
                       "dat_file": str(dat_file),
                       "covered_mutants_count": len(covered_by_this_test),
                       "killed_mutants": sorted(killed_by_this_test),
                       "skipped_mutants_count": len(already_killed_by_other_tests),
                       "survived_mutants_count": len(survived_by_this_test),
                       "analysis_start_time": str(analysis_timestamp_start),
                       "analysis_end_time": str(datetime.datetime.now())},
                      outfile, indent=2)

    print(f"Finished. Distinct mutants killed: {len(killed_mutants)} / "
          f"{mutation_tree.num_mutations} total; seeds tried: {invocations}.")


if __name__ == '__main__':
    main()
