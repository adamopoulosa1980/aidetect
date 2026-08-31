"""Build an EXAMPLE corpus. This is not a validation corpus.

It exists so that `aidetect-benchmark` can be run end to end by anyone, on data
anyone can regenerate, and so the shape of its output is visible without having
to assemble a corpus first. Numbers it produces are an illustration of the
output format. They are not evidence about how well anything detects.

The difference matters, so it is worth stating:

    Example corpus                    Validation corpus
    -----------------------------------------------------------------
    one small model on the AI side    the models people actually use
    one condition                     paraphrasing, non-native writers,
                                      Greek, several registers
    40 documents per class            hundreds per class
    abstracts, ~170 words             whole documents
    shows the harness runs            supports a claim

Human side: abstracts submitted to arXiv during 2010, years before any public
LLM, so authorship is not in question.

AI side: abstracts generated from the same titles by Qwen2.5-1.5B-Instruct.
Deliberately a different model family from the Falcon pair used for detection,
since scoring a model's output with its own relatives would flatter the result.
A 1.5B model also writes more predictably than the large ones people actually
use, which biases any result here optimistic.

Lengths are matched across the two sides, so nothing can be separated on length
alone.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NS = {"a": "http://www.w3.org/2005/Atom"}
MIN_WORDS, MAX_WORDS = 120, 260
WANTED = 40
CATEGORIES = ["cs.CL", "cs.LG", "math.ST", "physics.optics", "q-bio.NC"]

OUT = Path.home() / "arxiv2010-corpus"


def fetch_2010(category: str, count: int) -> list[tuple[str, str]]:
    query = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category} AND submittedDate:[201001010000 TO 201012312359]",
            "start": 0,
            "max_results": count,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    with urllib.request.urlopen(
        "http://export.arxiv.org/api/query?" + query, timeout=90
    ) as response:
        root = ET.fromstring(response.read().decode("utf-8"))

    out = []
    for entry in root.findall("a:entry", NS):
        title = " ".join(entry.find("a:title", NS).text.split())
        abstract = " ".join(entry.find("a:summary", NS).text.split())
        if MIN_WORDS <= len(abstract.split()) <= MAX_WORDS:
            out.append((title, abstract))
    return out


def generate(model, tok, title: str, target_words: int) -> str:
    """An abstract for ``title``, retried until it is long enough."""
    ask = (
        f"Write the abstract of an academic paper titled '{title}'. "
        f"Write approximately {target_words} words as a single paragraph of "
        "continuous prose. No headings, no bullet points, no title."
    )
    for attempt in range(3):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ask}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=int(target_words * 2.2) + 120,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        text = " ".join(text.split())
        if len(text.split()) >= MIN_WORDS:
            return text
        ask += " Write more detail and make it longer."
    return text


def main() -> None:
    human_dir = OUT / "arxiv-2010" / "human"
    ai_dir = OUT / "arxiv-2010" / "ai"
    human_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    papers: list[tuple[str, str]] = []
    for category in CATEGORIES:
        got = fetch_2010(category, 30)
        print(f"FETCH {category}: {len(got)} usable abstracts", flush=True)
        papers.extend(got)
        if len(papers) >= WANTED:
            break
    papers = papers[:WANTED]
    print(f"FETCH total {len(papers)} human abstracts from 2010", flush=True)

    for i, (_title, abstract) in enumerate(papers):
        (human_dir / f"{i:03d}.txt").write_text(abstract, encoding="utf-8")

    name = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

    for i, (title, abstract) in enumerate(papers):
        target = len(abstract.split())
        text = generate(model, tok, title, target)
        (ai_dir / f"{i:03d}.txt").write_text(text, encoding="utf-8")
        print(f"GEN {i + 1}/{len(papers)} {len(text.split())} words (target {target})", flush=True)

    human_lengths = [len(p.read_text(encoding="utf-8").split()) for p in human_dir.glob("*.txt")]
    ai_lengths = [len(p.read_text(encoding="utf-8").split()) for p in ai_dir.glob("*.txt")]
    print(f"DONE human n={len(human_lengths)} mean={sum(human_lengths) / len(human_lengths):.0f} words")
    print(f"DONE ai    n={len(ai_lengths)} mean={sum(ai_lengths) / len(ai_lengths):.0f} words")
    print(f"DONE corpus at {OUT}")


if __name__ == "__main__":
    main()
