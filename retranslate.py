#!/usr/bin/env python3
"""
Retranslate bilingual Arabic-English HTML pages using Claude API.
Processes pages 01.html–47.html sequentially with a rolling context
window so translations flow naturally across page breaks.

Usage:
    python3 retranslate.py              # process all pages
    python3 retranslate.py 5 10         # process pages 5–10 only (for reruns)
"""

import json
import re
import sys
import time
from pathlib import Path
from bs4 import BeautifulSoup
import anthropic

PAGES_DIR = Path(__file__).parent
NUM_PAGES = 47
CONTEXT_TAIL = 4      # non-empty translated blocks to carry forward as context
MAX_RETRIES = 3
RETRY_DELAY = 5       # seconds between retries

MODEL = "claude-opus-4-8"

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are translating "المجالس العاشورية في المآتم الحسينية" \
(The Ashura Gatherings in the Husayni Mourning Assemblies) \
by Sheikh Abdullah ibn al-Haj Hasan Al-Darwish from Arabic into English.

Guidelines:
- Audience: general public. Use clear, natural, flowing English — not overly scholarly.
- Preserve the emotional weight, reverence, and gravity of the original.
- Standard English names: Hussein, Hassan, Zainab, Fatima, Ali, Umar, Yazid, etc.
- Use "(peace be upon him)", "(peace be upon her)", "(peace be upon them)" for religious figures exactly as in Islamic tradition.
- If context from the previous page is provided, continue seamlessly — do NOT restart cold.
- If a block's Arabic begins mid-sentence (continuing from the previous page), complete it naturally.
- For poetry blocks (marked is_poetry=true): translate line by line, one translated line per Arabic line, preserving the lyrical feel.
- Keep source citations exactly as they appear (e.g. "Bihar al-Anwar, al-Majlisi: 44/278, Hadith 2").
- Empty blocks (empty arabic string) → return empty string "".
- Output ONLY a valid JSON array of strings, one per input block, in order. No explanation, no markdown fences."""


def get_arabic_lines(div):
    """For poetry blocks, return list of Arabic lines (one per <nobr>)."""
    return [n.get_text(strip=True) for n in div.find_all('nobr')]


def extract_blocks(html_path):
    """Parse an HTML file and return a list of block dicts."""
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')
    blocks = []
    for i, bdiv in enumerate(soup.find_all('div', class_='bilingual')):
        is_poetry = 'couplets' in bdiv.get('class', [])
        ar_div = bdiv.find('div', class_='arabic')
        tr_div = bdiv.find('div', class_='translation')

        if is_poetry and ar_div:
            arabic_text = '\n'.join(get_arabic_lines(ar_div))
        else:
            arabic_text = ar_div.get_text(separator='\n', strip=True) if ar_div else ''

        # Normalise whitespace runs but keep intentional newlines
        arabic_text = re.sub(r'\n{3,}', '\n\n', arabic_text).strip()

        blocks.append({
            'index': i,
            'arabic': arabic_text,
            'is_poetry': is_poetry,
        })
    return blocks


def translate_page(page_num, blocks, context_tail):
    """Call Claude API and return a list of translated strings (same length as blocks).

    Only non-empty blocks are sent to the API; empty blocks get '' automatically.
    Results are stitched back to match the full blocks list.
    """
    context_section = ''
    if context_tail:
        joined = '\n---\n'.join(context_tail)
        context_section = (
            f"CONTEXT — last translated paragraphs from the previous page "
            f"(maintain continuity, do not repeat them):\n{joined}\n\n"
        )

    # Filter to only blocks that have Arabic text
    non_empty = [(i, b) for i, b in enumerate(blocks) if b['arabic'].strip()]

    if not non_empty:
        return ['' for _ in blocks]

    input_data = [
        {'seq': seq, 'arabic': b['arabic'], 'is_poetry': b['is_poetry']}
        for seq, (_, b) in enumerate(non_empty)
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
            if len(api_results) != len(non_empty):
                raise ValueError(
                    f"Expected {len(non_empty)} strings, got {len(api_results)}"
                )

            # Stitch back: fill empty slots with '', non-empty with API result
            full = ['' for _ in blocks]
            for seq, (block_idx, _) in enumerate(non_empty):
                full[block_idx] = api_results[seq] if seq < len(api_results) else ''
            return full

        except Exception as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def inject_translations(html_path, translations):
    """Write new translation strings back into the .translation divs in-place."""
    content = html_path.read_text(encoding='utf-8')
    soup = BeautifulSoup(content, 'html.parser')
    bilingual_divs = soup.find_all('div', class_='bilingual')

    changed = 0
    for i, bdiv in enumerate(bilingual_divs):
        if i >= len(translations):
            break
        tr_div = bdiv.find('div', class_='translation')
        if tr_div is None:
            continue
        new_text = translations[i] if i < len(translations) else ''
        if new_text is None:
            new_text = ''

        # Convert newlines → <br> for display; double newlines → <br><br>
        new_html = (
            new_text
            .replace('\n\n', '<br><br>')
            .replace('\n', '<br>')
        )
        tr_div.clear()
        if new_html:
            tr_div.append(BeautifulSoup(new_html, 'html.parser'))
        changed += 1

    html_path.write_text(str(soup), encoding='utf-8')
    return changed


def main():
    start_page = 1
    end_page = NUM_PAGES
    if len(sys.argv) == 3:
        start_page, end_page = int(sys.argv[1]), int(sys.argv[2])
    elif len(sys.argv) == 2:
        start_page = end_page = int(sys.argv[1])

    rolling_context = []  # last N non-empty translated strings

    for page_num in range(start_page, end_page + 1):
        filename = f"{page_num:02d}.html"
        html_path = PAGES_DIR / filename

        if not html_path.exists():
            print(f"[{page_num:02d}] SKIP — file not found")
            continue

        print(f"[{page_num:02d}/{NUM_PAGES}] Translating {filename} ...", flush=True)

        blocks = extract_blocks(html_path)
        non_empty = sum(1 for b in blocks if b['arabic'].strip())
        print(f"       {len(blocks)} blocks total, {non_empty} with Arabic text", flush=True)

        try:
            translations = translate_page(page_num, blocks, rolling_context)
        except Exception as e:
            print(f"       ERROR — skipping page: {e}", file=sys.stderr)
            continue

        # Pad to match block count if model returned fewer
        while len(translations) < len(blocks):
            translations.append('')

        inject_translations(html_path, translations)

        # Update rolling context with last N non-empty translations
        new_non_empty = [t for t in translations if isinstance(t, str) and t.strip()]
        rolling_context = new_non_empty[-CONTEXT_TAIL:]

        done = sum(1 for t in translations if isinstance(t, str) and t.strip())
        print(f"       Done — {done} blocks translated, {len(rolling_context)} carried to next page")

    print("\nAll done.")


if __name__ == '__main__':
    main()
