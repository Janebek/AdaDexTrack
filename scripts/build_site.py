"""Build the existing project website without including simulator resources."""

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ('index.html', 'manifest.json', 'CNAME', 'robots.txt', 'favicon.ico')
PUBLIC_DIRS = ('static', 'videos', 'thumbs')


def build_site(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    for name in ('index.html', 'manifest.json'):
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    if destination == source or destination in source.parents:
        raise ValueError('The website output must be a new build directory')
    if any(source/name == destination or source/name in destination.parents for name in PUBLIC_DIRS):
        raise ValueError('The output cannot be inside website input directories')
    destination.mkdir(parents=True, exist_ok=False)
    for name in PUBLIC_FILES:
        if (source/name).is_file():
            shutil.copyfile(str(source/name), str(destination/name))
    for name in PUBLIC_DIRS:
        if (source/name).is_dir():
            shutil.copytree(str(source/name), str(destination/name))
    (destination/'.nojekyll').touch()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'_site')
    args = parser.parse_args()
    print('[WEBSITE] Built '+str(build_site(ROOT, args.output_dir)))


if __name__ == '__main__':
    main()
