"""Readable, unique names shared by the CLI and browser interface."""
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def automatic_output(image, query, root=None):
    def slug(text, limit, fallback):
        return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:limit].rstrip('-') or fallback
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]/'outputs'
    name = '__'.join((slug(Path(image).stem, 24, 'image'),
                      slug(query, 36, 'query'),
                      datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8]))
    return root/name
