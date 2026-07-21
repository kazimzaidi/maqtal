#!/usr/bin/env python3
"""
Build Arabic-Urdu bilingual pages under urdu/ from the existing 01.html-47.html.

The Arabic text is re-extracted from the source pages (source of truth) and
translated fresh into Urdu with Claude — NOT pivoted through the English
translation, to avoid compounding drift. The English pages are otherwise used
purely as a structural template (nav, page markers, arabic markup), since that
structure is identical for the Urdu edition.

Usage:
    python3 urdu_translate.py              # process all pages
    python3 urdu_translate.py 5 10         # process pages 5-10 only (for reruns)
    python3 urdu_translate.py 20 20        # single page, for a quality spot-check
"""

import json
import re
import sys
import time
from pathlib import Path
from bs4 import BeautifulSoup
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from retranslate import extract_blocks  # reuse Arabic block extraction

PAGES_DIR = Path(__file__).parent
URDU_DIR = PAGES_DIR / "urdu"
NUM_PAGES = 47
CONTEXT_TAIL = 4
MAX_RETRIES = 3
RETRY_DELAY = 5

MODEL = "claude-opus-4-8"
CHUNK_SIZE = 10   # blocks per API call — keep output well under max_tokens

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are translating "المجالس العاشورية في المآتم الحسينية" \
(The Ashura Gatherings in the Husayni Mourning Assemblies) \
by Sheikh Abdullah ibn al-Haj Hasan Al-Darwish from Arabic into Urdu.

Guidelines:
- Audience: general Urdu-speaking public. Use clear, natural, flowing Urdu prose — not overly literal or stilted, but still dignified and suited to religious content.
- Preserve the emotional weight, reverence, and gravity of the original.
- Use the honorifics conventional in Urdu Shia religious literature, in Arabic script exactly as customarily printed: "(علیہ السلام)", "(علیہا السلام)", "(علیہم السلام)", "(صلی اللہ علیہ وآلہ)".
- If context from the previous page is provided, continue seamlessly — do NOT restart cold.
- If a block's Arabic begins mid-sentence (continuing from the previous page), complete it naturally.
- For poetry blocks (marked is_poetry=true): translate line by line, one translated line per Arabic line, preserving the lyrical and elegiac register where possible.
- Render source citations (book title, volume/page, hadith number) in Urdu, keeping the same structure as the original.
- Empty blocks (empty arabic string) -> return empty string "".
- Output ONLY a valid JSON array of strings, one per input block, in order. No explanation, no markdown fences."""

TITLE_SYSTEM_PROMPT = """Translate short Arabic book-title phrases into Urdu, \
in the dignified register used on the cover of Shia religious books. \
Output ONLY a JSON array of strings, same order, no explanation."""


def translate_chunk(page_num, chunk_blocks, context_tail):
    """Translate a single chunk of (arabic, is_poetry) blocks. Returns list of strings."""
    context_section = ''
    if context_tail:
        joined = '\n---\n'.join(context_tail)
        context_section = (
            f"CONTEXT — last translated paragraphs from the previous chunk/page "
            f"(maintain continuity, do not repeat them):\n{joined}\n\n"
        )

    input_data = [
        {'seq': seq, 'arabic': b['arabic'], 'is_poetry': b['is_poetry']}
        for seq, b in enumerate(chunk_blocks)
    ]

    user_msg = (
        f"{context_section}"
        f"Translate the following {len(input_data)} blocks from page {page_num}.\n\n"
        f"{json.dumps(input_data, ensure_ascii=False, indent=2)}\n\n"
        f"Return a JSON array of exactly {len(input_data)} strings (seq 0 to {len(input_data)-1})."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            match = re.search(r'\[[\s\S]*\]', raw)
            if not match:
                raise ValueError(f"No JSON array in response:\n{raw[:300]}")
            api_results = json.loads(match.group())
            if not isinstance(api_results, list):
                raise ValueError("Response is not a JSON list")
            if len(api_results) != len(chunk_blocks):
                raise ValueError(
                    f"Expected {len(chunk_blocks)} strings, got {len(api_results)}"
                )
            return api_results

        except Exception as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def translate_page(page_num, blocks, context_tail):
    """Call Claude API in chunks and return a list of translated strings (same length as blocks)."""
    non_empty = [(i, b) for i, b in enumerate(blocks) if b['arabic'].strip()]

    if not non_empty:
        return ['' for _ in blocks]

    full = ['' for _ in blocks]
    running_context = list(context_tail)

    for start in range(0, len(non_empty), CHUNK_SIZE):
        chunk = non_empty[start:start + CHUNK_SIZE]
        chunk_blocks = [b for _, b in chunk]
        results = translate_chunk(page_num, chunk_blocks, running_context)

        for (block_idx, _), text in zip(chunk, results):
            full[block_idx] = text if text is not None else ''

        new_non_empty = [t for t in results if isinstance(t, str) and t.strip()]
        if new_non_empty:
            running_context = new_non_empty[-CONTEXT_TAIL:]

    return full


def translate_title():
    """One-off translation of the book title/subtitle line, reused on every page."""
    subtitle = "The ʿĀshūrāʾ gatherings in the Ḥusaynī mourning assemblies"
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=TITLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps([subtitle], ensure_ascii=False)}],
    )
    raw = response.content[0].text.strip()
    match = re.search(r'\[[\s\S]*\]', raw)
    result = json.loads(match.group())
    return result[0]


def build_urdu_page(page_num, translations):
    """Write urdu/NN.html using the English page as a structural template."""
    filename = f"{page_num:02d}.html"
    src = (PAGES_DIR / filename).read_text(encoding='utf-8')
    soup = BeautifulSoup(src, 'html.parser')

    html_tag = soup.find('html')
    if html_tag:
        html_tag['lang'] = 'ur'

    bilingual_divs = soup.find_all('div', class_='bilingual')
    for i, bdiv in enumerate(bilingual_divs):
        tr_div = bdiv.find('div', class_='translation')
        if tr_div is None:
            continue
        new_text = translations[i] if i < len(translations) else ''
        if new_text is None:
            new_text = ''
        new_html = new_text.replace('\n\n', '<br><br>').replace('\n', '<br>')
        tr_div.clear()
        tr_div['dir'] = 'rtl'
        if new_html:
            tr_div.append(BeautifulSoup(new_html, 'html.parser'))

    URDU_DIR.mkdir(exist_ok=True)
    (URDU_DIR / filename).write_text(str(soup), encoding='utf-8')


def main():
    start_page = 1
    end_page = NUM_PAGES
    if len(sys.argv) == 3:
        start_page, end_page = int(sys.argv[1]), int(sys.argv[2])
    elif len(sys.argv) == 2:
        start_page = end_page = int(sys.argv[1])

    rolling_context = []

    for page_num in range(start_page, end_page + 1):
        filename = f"{page_num:02d}.html"
        html_path = PAGES_DIR / filename

        if not html_path.exists():
            print(f"[{page_num:02d}] SKIP — file not found")
            continue

        print(f"[{page_num:02d}/{NUM_PAGES}] Translating {filename} to Urdu ...", flush=True)

        blocks = extract_blocks(html_path)
        non_empty = sum(1 for b in blocks if b['arabic'].strip())
        print(f"       {len(blocks)} blocks total, {non_empty} with Arabic text", flush=True)

        try:
            translations = translate_page(page_num, blocks, rolling_context)
        except Exception as e:
            print(f"       ERROR — skipping page: {e}", file=sys.stderr)
            continue

        while len(translations) < len(blocks):
            translations.append('')

        build_urdu_page(page_num, translations)

        new_non_empty = [t for t in translations if isinstance(t, str) and t.strip()]
        rolling_context = new_non_empty[-CONTEXT_TAIL:]

        done = sum(1 for t in translations if isinstance(t, str) and t.strip())
        print(f"       Done — {done} blocks translated, {len(rolling_context)} carried to next page")

    print("\nAll done.")


if __name__ == '__main__':
    main()
