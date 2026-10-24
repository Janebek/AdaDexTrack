"""Import all paper references and their original splits into a portable dataset.

The source repository is needed only for this one-time import. Reference bytes
are copied unchanged. Existing destination datasets and differing assets are
never replaced. No checkpoint filtering or height filtering is applied.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adadextrack.data.reference import load_reference
from adadextrack.data.dataset import read_manifest

FIELDS = ('reference_id', 'group_id', 'object_name', 'reference_file',
          'num_frames', 'description', 'sha256')
SPLITS = {
    'unseen_trajectory_train': ('train', 2212),
    'unseen_trajectory_test': ('test', 553),
    'unseen_object_train': ('object_train', 2520),
    'unseen_object_test': ('object_test', 245),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_resource(source, destination):
    """Add missing files; reject any inconsistent existing resource."""
    if destination.exists():
        if not destination.is_file() or digest(source) != digest(destination):
            raise ValueError('Existing resource differs from source: '+str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(destination))
    if digest(source) != digest(destination):
        raise ValueError('Resource copy checksum mismatch: '+str(destination))


def build(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    metadata_dir = source/'data/diffh2o/10_traj_per_teacher'
    object_map = json.loads((metadata_dir/'label_to_object.json').read_text())
    teachers = json.loads((metadata_dir/'teacher_manifests/teachers.json').read_text())
    membership = {}
    for teacher in teachers:
        if len(teacher['labels']) != len(teacher['p_list']):
            raise ValueError('Invalid teacher membership')
        for label, paraphrase in zip(teacher['labels'], teacher['p_list']):
            label = str(int(label))
            if label in membership:
                raise ValueError('Duplicate source teacher membership: '+label)
            membership[label] = (teacher, paraphrase)
    splits, source_checksums = {}, {}
    for name, (suffix, expected) in SPLITS.items():
        path = source/('debug/evaluation_summary_10refs_filtered_10cm_60deg_'+suffix+'.csv')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        labels = [str(int(row['label'])) for row in rows]
        if len(labels) != expected or len(set(labels)) != expected:
            raise ValueError('Wrong coverage or duplicates in '+name)
        splits[name] = labels
        source_checksums[name] = dict(filename=path.name, sha256=digest(path), references=expected)
    full = set(splits['unseen_trajectory_train']) | set(splits['unseen_trajectory_test'])
    if len(full) != 2765:
        raise ValueError('Expected 2765 distinct references')
    for prefix in ('unseen_trajectory', 'unseen_object'):
        train, test = map(set, (splits[prefix+'_train'], splits[prefix+'_test']))
        if train & test or train | test != full:
            raise ValueError('Splits must partition the same full dataset: '+prefix)
    labels = sorted(full, key=int)
    objects = sorted({object_map[label] for label in labels})
    if len(objects) != 50:
        raise ValueError('Expected 50 objects')
    bases = sorted({membership[label][0]['base'] for label in labels},
                   key=lambda base: (object_map[next(label for label in labels
                       if membership[label][0]['base'] == base)], base))
    group_ids = {base: index for index, base in enumerate(bases)}
    # Validate all numerical references before creating a destination dataset.
    paths = {}
    for label in labels:
        path = source/'debug/diffh2o_generated_5050_Retar2'/('leap_passive_active_info_ori_diffh2o_'+label+'_nf_300.npy')
        load_reference(path)
        paths[label] = path
    latents = {path.name.lower(): path for path in (source/'utils/latents_out').glob('*.npy')}
    for obj in objects:
        folder = source/'assets/urdf/objects/GRAB'/obj
        if Path(obj).name != obj or obj in ('.', '..'):
            raise ValueError('Invalid object name')
        for mesh in ET.parse(str(folder/(obj+'.urdf'))).iter('mesh'):
            mesh_path = (folder/mesh.attrib['filename']).resolve()
            if folder.resolve() not in mesh_path.parents or not mesh_path.is_file():
                raise ValueError('Missing or external mesh dependency: '+str(mesh_path))
        latent = latents[obj.lower()+'_latent.npy']
        array = np.load(str(latent), allow_pickle=False)
        if array.shape != (64,) or not np.isfinite(array).all():
            raise ValueError('Invalid latent: '+obj)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                copy_resource(path, ROOT/'assets/urdf/objects/GRAB'/obj/path.relative_to(folder))
        copy_resource(latent, ROOT/'assets/latents'/latent.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=destination.name+'.building-', dir=str(destination.parent)))
    try:
        for name in ('references', 'splits', 'groups', 'provenance'):
            (staged/name).mkdir()
        manifest, mapping, by_label = [], [], {}
        for ref_id, label in enumerate(labels):
            teacher, paraphrase = membership[label]
            group_id = group_ids[teacher['base']]
            relative = 'references/ref_{:04d}.npy'.format(ref_id)
            shutil.copyfile(str(paths[label]), str(staged/relative))
            checksum = digest(paths[label])
            if digest(staged/relative) != checksum:
                raise ValueError('Reference copy checksum mismatch: '+label)
            row = dict(reference_id=ref_id, group_id=group_id, object_name=object_map[label],
                       reference_file=relative, num_frames=300, description=teacher['base'], sha256=checksum)
            manifest.append(row)
            by_label[label] = row
            mapping.append(dict(reference_id=ref_id, group_id=group_id, source_reference_label=label,
                source_teacher_id=teacher['teacher_id'], source_semantic_group=teacher['base'],
                source_paraphrase_index=paraphrase, source_reference_filename=paths[label].name))
        write_csv(staged/'manifest.csv', FIELDS, manifest)
        for name, members in splits.items():
            write_csv(staged/'splits'/(name+'.csv'), FIELDS, [by_label[label] for label in members])
        groups = []
        for base, group_id in group_ids.items():
            members = [row for row in manifest if row['group_id'] == group_id]
            if len({row['object_name'] for row in members}) != 1:
                raise ValueError('Semantic group contains different objects')
            groups.append(dict(group_id=group_id, object_name=members[0]['object_name'],
                reference_count=len(members), reference_ids=json.dumps([r['reference_id'] for r in members]), description=base))
            write_csv(staged/'groups'/('group_{:03d}.csv'.format(group_id)), FIELDS, members)
        write_csv(staged/'semantic_groups.csv', list(groups[0]), groups)
        write_csv(staged/'provenance/source_mapping.csv', list(mapping[0]), mapping)
        write_json(staged/'provenance/import.json', dict(source_root=str(source), source_splits=source_checksums,
            filtering='Original paper CSV membership only; no height, BC success, or accuracy filtering.',
            description_note='Descriptions retain the source semantic base names; natural-language captions are not imported.'))
        write_json(staged/'dataset.json', dict(schema_version=1, name='AdaDexTrack-Full', references=len(manifest),
            semantic_groups=len(groups), objects=len(objects), frames_per_reference=300,
            manifest='manifest.csv', group_index='semantic_groups.csv', group_manifests='groups/group_{group_id:03d}.csv',
            splits={name:dict(manifest='splits/'+name+'.csv', references=len(members)) for name,members in splits.items()},
            reference_id_range=[0,len(manifest)-1], group_id_range=[0,len(groups)-1],
            id_assignment='Full-dataset IDs assigned by ascending source label. Split manifests retain global IDs and original CSV order. IDs differ from the demonstration subsets.',
            path_rule='reference_file is relative to the ancestor containing dataset.json.',
            units=dict(position='meter', joint_angle='radian', quaternion_order='xyzw'),
            reference_format='Original 300-frame numpy dictionaries copied byte-for-byte.',
            provenance='provenance/source_mapping.csv', object_counts=dict(sorted(Counter(r['object_name'] for r in manifest).items())),
            purpose='Full original paper reference corpus with both original train/test partitions. No checkpoint-based selection.'))
        read_manifest(staged/'manifest.csv', verify_hashes=True)
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)
    except BaseException:
        shutil.rmtree(str(staged))
        raise
    print(json.dumps(dict(dataset=str(destination), references=len(manifest), objects=len(objects),
        semantic_groups=len(groups), splits={name:len(members) for name,members in splits.items()}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/adadextrack_full')
    args = parser.parse_args()
    build(args.source_root.expanduser().resolve(), args.output_dir.expanduser().resolve())


if __name__ == '__main__':
    main()
