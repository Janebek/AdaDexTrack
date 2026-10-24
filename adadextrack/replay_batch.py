"""Sequential group or reference-batch replay in fresh Isaac Gym processes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from adadextrack.data.groups import partition_groups

ROOT = Path(__file__).resolve().parents[1]


def partition_batches(entries, batch_size):
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    partition_groups(entries)  # Reject empty selections, duplicate IDs and inconsistent group metadata.
    return [entries[start:start + batch_size] for start in range(0, len(entries), batch_size)]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def load_worker_args(config_path):
    values = json.loads(config_path.read_text())
    for key in ('manifest', 'output', 'snapshot'):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return argparse.Namespace(**values)


def verify_recording(report_path, output, expected_ids, expected_frames):
    report = json.loads(report_path.read_text())
    if report['references'] != len(expected_ids) or report['status'] not in ('complete', 'cancelled'):
        raise ValueError('Invalid replay report')
    frames = report['frames']
    if (report['status'] == 'complete' and frames != expected_frames
            or not 0 <= frames <= expected_frames):
        raise ValueError('Incomplete replay recording')
    with np.load(str(output), allow_pickle=False) as archive:
        if archive['reference_ids'].tolist() != expected_ids:
            raise ValueError('Replay reference IDs or ordering differ')
        for key, width in (('obj_poses', 7), ('robot_qpos', 22)):
            array = archive[key]
            if array.shape != (frames, len(expected_ids), width) or not np.isfinite(array).all():
                raise ValueError('Invalid replay states: '+key)
        if not np.array_equal(archive['reference_frames'], np.arange(frames) % 300):
            raise ValueError('Invalid replay frame alignment')
    errors = [float(report[key]) for key in ('max_object_state_error', 'max_joint_error')]
    if not all(np.isfinite(error) and 0 <= error <= 1e-4 for error in errors):
        raise ValueError('Replay reference states changed')
    return report


def run_groups(args, entries):
    return run_partitions(args, partition_groups(entries), 'group')


def run_batches(args, entries):
    return run_partitions(args, partition_batches(entries, args.batch_size), 'batch')


def run_partitions(args, partitions, mode):
    key = 'groups' if mode == 'group' else 'batches'
    entries = [entry for selection in partitions for entry in selection]
    if args.output_dir is not None:
        run_dir = args.output_dir.expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
        run_dir = ROOT/'outputs/replay'/stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.manifest.expanduser().resolve()
    summary = dict(status='running', manifest=str(manifest),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        requested_references=len(entries), completed_references=0, loops=args.loops,
        expected_frames_per_reference=300*args.loops, headless=args.headless,
        mode='kinematic_reference_replay', playback_mode=mode,
        reference_ids=[entry.reference_id for entry in entries])
    summary.update({'requested_'+key: len(partitions), 'completed_'+key: 0, key: []})
    if mode == 'batch':
        summary['batch_size'] = args.batch_size
    summary_path = run_dir/'summary.json'
    write_json(summary_path, summary)
    print('[REPLAY] {} references in {} {}. Results: {}'.format(
        len(entries), len(partitions), key, run_dir), flush=True)
    process = None
    try:
        for index, selection in enumerate(partitions):
            group_id = int(selection[0].group_id) if mode == 'group' else None
            number = group_id if mode == 'group' else index
            name = '{}_{:03d}'.format(mode, number)
            directory = run_dir/name
            directory.mkdir()
            output = args.output.expanduser().resolve() if args.output is not None else directory/'reference_replay.npz'
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise FileExistsError(output)
            config = dict(manifest=str(manifest), reference_id=[int(entry.reference_id) for entry in selection],
                group_id=group_id, mode=mode, all=False, overview=args.overview, headless=args.headless,
                device=args.device, loops=args.loops, fps=args.fps, snapshot_frame=args.snapshot_frame,
                output=str(output), snapshot=None)
            if args.snapshot is not None:
                snapshot = args.snapshot.expanduser().resolve()
                config['snapshot'] = str(snapshot.with_name(snapshot.stem+'_'+name+snapshot.suffix))
            config_path = directory/'config.json'
            write_json(config_path, config)
            print('[REPLAY] {} ({}/{}): {} references.'.format(name,index+1,len(partitions),len(selection)),flush=True)
            with (directory/'worker.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-u', '-B', str(ROOT/'scripts/replay.py'),
                    '--_group-config', str(config_path)], cwd=str(ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
                process = None
                if code != 0:
                    raise RuntimeError('Replay {} failed (exit {}). See {}'.format(name,code,directory/'worker.log'))
            report = verify_recording(output.with_suffix('.json'), output,
                [entry.reference_id for entry in selection], 300*args.loops)
            record = dict(reference_ids=config['reference_id'], **report)
            record.update({'group_id': group_id} if mode == 'group' else {'batch_index': index})
            summary[key].append(record)
            if report['status'] == 'cancelled':
                summary['status'] = 'cancelled'
                write_json(summary_path, summary)
                print('[REPLAY] Viewer closed; remaining {} will not be opened. Results: {}'.format(key,run_dir),flush=True)
                return 0
            summary['completed_references'] += len(selection)
            summary['completed_'+key] += 1
            write_json(summary_path, summary)
        actual_ids = [str(ref_id) for recording in summary[key] for ref_id in recording['reference_ids']]
        if actual_ids != summary['reference_ids']:
            raise ValueError('Replay did not cover the selection exactly')
        summary['status'] = 'complete'
        summary['max_object_state_error'] = max(recording['max_object_state_error'] for recording in summary[key])
        summary['max_joint_error'] = max(recording['max_joint_error'] for recording in summary[key])
        write_json(summary_path, summary)
        print('[REPLAY] Completed all {} references in {} {}. Results: {}'.format(
            len(entries),len(partitions),key,run_dir),flush=True)
        return 0
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        summary['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        summary['error'] = str(error)
        write_json(summary_path, summary)
        raise
