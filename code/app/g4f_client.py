from typing import List
from g4f.client import Client


def _first_choice_text(resp) -> str:
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def g4f_complete(messages: List[dict], model: str) -> str:
    client = Client()
    resp = client.chat.completions.create(model=model, messages=messages)
    return _first_choice_text(resp)


def g4f_batch_rewrite(lines: List[str], model: str, lang_hint: str = "") -> List[str]:
    if not lines:
        return []
    sep = "|||"
    payload = f"{sep}".join([l.replace('\n', ' ').strip() for l in lines])
    sys = "You are a precise text corrector. Fix recognition mistakes by context, keep the same language, style and meaning. Do not add or remove information. Preserve line count and order. Return exactly the same number of lines, separated by newline only."
    if lang_hint:
        sys += f" The language of the lines is {lang_hint}."
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user",
         "content": f"Input lines separated by {sep}:\n{payload}\nReturn corrected lines as raw text with newline separators only."}
    ]
    out = g4f_complete(msgs, model)
    parts = [p.strip() for p in out.splitlines() if p.strip() != ""]
    if len(parts) != len(lines):
        return lines
    return parts


def g4f_batch_translate(lines: List[str], model: str, target_lang: str, source_lang: str = "auto") -> List[str]:
    if not lines:
        return []
    clean = [l.replace('\n', ' ').strip() for l in lines]
    joined = "\n".join(clean)
    sys = "You are a professional subtitle translator. Translate lines faithfully, concise, natural, keep timing segmentation implicit by preserving line boundaries. Do not add numbering, metadata, or quotes. Output exactly the same number of lines in the same order."
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user",
         "content": f"Source language: {source_lang}. Target language: {target_lang}.\nTranslate each of these lines one-by-one, outputting the translations as newline-separated lines only:\n{joined}"}
    ]
    out = g4f_complete(msgs, model)
    parts = [p.strip() for p in out.splitlines()]
    if len(parts) != len(clean):
        return clean
    return parts
