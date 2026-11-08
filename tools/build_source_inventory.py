"""Index local research evidence without publishing its third-party contents."""
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'literature' / '2026-11-06'


def entry(path):
    return {
        'path': path.relative_to(ROOT).as_posix(),
        'size_bytes': path.stat().st_size,
        'sha256': sha256(path.read_bytes()).hexdigest(),
    }


def main():
    records = json.loads((REVIEW / 'paper_registry.json').read_text(encoding='utf-8'))['records']
    sources = REVIEW / 'sources'
    files = sorted(p for p in sources.iterdir() if p.is_file() and p.name != 'README.md')
    for path in [ROOT / 'IRDS' / 'IRDS.pdf', ROOT / 'KIMORE' / 'KIMORE.pdf']:
        if path.exists():
            files.append(path)
    inventory = {
        'generated_on': date.today().isoformat(),
        'research_cutoff': '2026-11-06',
        'note': 'Local evidence inventory only; hashes do not verify research conclusions. Cached contents are not distributed.',
        'files': [entry(p) for p in files],
        'papers': [
            {k: r.get(k) for k in ['id', 'title', 'doi', 'url', 'evidence_level', 'metadata_source', 'source_stem']}
            for r in records
        ],
        'resource_urls': [
            'https://bsu3dvlab.github.io/FineRehab/',
            'https://zenodo.org/records/13305826',
            'https://github.com/nguyensmai/KeraalDataset',
            'https://github.com/MSD-IRIMAS/DeepRehabPile',
            'https://github.com/shamarkus/ReCon',
            'https://github.com/bruceyo/EGCN',
            'https://doi.org/10.7910/DVN/XSI0QN',
        ],
    }
    target = REVIEW / 'source_inventory.json'
    target.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Indexed {len(files)} local evidence files and {len(records)} papers.')


if __name__ == '__main__':
    main()
