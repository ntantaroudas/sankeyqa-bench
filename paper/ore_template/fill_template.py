"""Fill the official Open Research Europe Article Template with the SankeyQA manuscript.

    python paper/ore_template/fill_template.py paper/docx/manuscript.json paper/SankeyQA_ORE_manuscript.docx

Adapted from the authors' KyGround build (same template, same method).

The template (ORE_Article_Template.docx, F1000/ORE, created 2022-12-14, retrieved from ORE's own link through the
Internet Archive copy of 2024-04-20) is edited at the XML level: its styles, theme, settings, header ("https://open-
research-europe.ec.europa.eu | editorial@...") and footer ("Information Classification: General", "Page X of Y"),
page set-up, numbering and custom properties are kept byte for byte, except that

  * word/document.xml keeps the template's first paragraph ("Open Research Europe Article Template"), the empty
    paragraph after it and the section properties; the instruction text in between is replaced by the manuscript,
    written with the template's own styles (Heading1 for section labels, Heading2 for subsections, BodyText for text,
    ListParagraph with the template's bullet numbering for lists, the template's PlainTable1 table style for tables,
    bold "Table n." / "Figure n." labels with the caption below the table or figure, as in the template's examples);
  * word/numbering.xml gains one decimal list (the template has bullets and a "[n]" list only);
  * word/_rels/document.xml.rels loses the relationships of the removed instruction text (hyperlinks to ORE's
    guidelines and the ORE logo used as a sample figure) and gains the manuscript's hyperlinks and figures;
  * docProps/core.xml carries the article's title and authors instead of the template's.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "ORE_Article_Template.docx"

PAGE_W, MARGIN_L, MARGIN_R = 11910, 920, 920          # from the template's sectPr (A4)
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R               # 10070 DXA
FIG_W_EMU = int(6.3 * 914400)                          # figures at their own size, at most 6.3 in wide (text 6.99 in)
BODY_SZ, TABLE_SZ, NOTE_SZ, TITLE_SZ = 19, 17, 17, 24  # half-points (BodyText is 9.5 pt)
NEW_PAGE = {"Figure legends", "Tables"}
BULLET_NUMID, DECIMAL_NUMID = 3, 4                     # 3 = the template's bullet list; 4, 5, ... are added below,
                                                       # one per numbered list, so that each list starts at 1

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_HYPERLINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
REL_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


class Doc:
    def __init__(self) -> None:
        self.rels: list[tuple[str, str, str, bool]] = []    # (id, type, target, external)
        self.links: dict[str, str] = {}
        self.media: list[tuple[str, bytes]] = []
        self.next_rid = 101
        self.next_pic = 101
        self.decimal_lists = 0                          # numbered lists, each with its own numbering instance

    def rid(self) -> str:
        r = f"rId{self.next_rid}"
        self.next_rid += 1
        return r

    def link(self, url: str) -> str:
        if url not in self.links:
            r = self.rid()
            self.links[url] = r
            self.rels.append((r, REL_HYPERLINK, url, True))
        return self.links[url]

    def image(self, data: bytes) -> str:
        n = len(self.media) + 1
        name = f"media/sankeyqa_figure{n}.png"
        self.media.append((f"word/{name}", data))
        r = self.rid()
        self.rels.append((r, REL_IMAGE, name, False))
        return r


# ------------------------------------------------------------------------------------------------ runs

def t_xml(text: str) -> str:
    parts = text.split("\t")
    out = []
    for k, p in enumerate(parts):
        if k:
            out.append("<w:tab/>")
        if p:
            out.append(f'<w:t xml:space="preserve">{escape(p)}</w:t>')
    return "".join(out)


def rpr(r: dict, size: int | None = None, bold: bool = False, link: bool = False) -> str:
    x = []
    if link:
        x.append('<w:rStyle w:val="Hyperlink"/>')
    if r.get("code"):
        x.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>')
    if r.get("b") or bold:
        x.append("<w:b/><w:bCs/>")
    if r.get("i"):
        x.append("<w:i/><w:iCs/>")
    sz = size
    if r.get("code"):
        sz = (size or BODY_SZ) - 2
    if sz:
        x.append(f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>')
    if r.get("hl"):                                    # text the authors still have to complete
        x.append('<w:shd w:val="clear" w:color="auto" w:fill="FFF200"/>')
    if r.get("sup"):
        x.append('<w:vertAlign w:val="superscript"/>')
    return f"<w:rPr>{''.join(x)}</w:rPr>" if x else ""


def runs_xml(doc: Doc, runs: list[dict], size: int | None = None, bold: bool = False) -> str:
    out = []
    for r in runs:
        if r.get("link"):
            rid = doc.link(r["link"])
            out.append(f'<w:hyperlink r:id="{rid}" w:history="1"><w:r>{rpr(r, size, bold, link=True)}'
                       f'{t_xml(r["t"])}</w:r></w:hyperlink>')
        else:
            out.append(f"<w:r>{rpr(r, size, bold)}{t_xml(r['t'])}</w:r>")
    return "".join(out)


def para(doc: Doc, runs: list[dict], style: str = "BodyText", *, keep_next=False, keep_lines=False,
         page_break=False, num: tuple[int, int] | None = None, tabs: list[int] = (), before=None, after=None,
         line=None, ind: dict | None = None, jc: str | None = None, mark_sz: int | None = None,
         size: int | None = None, bold: bool = False, raw_runs: str = "") -> str:
    p = [f'<w:pStyle w:val="{style}"/>']
    if keep_next:
        p.append("<w:keepNext/>")
    if keep_lines:
        p.append("<w:keepLines/>")
    if page_break:
        p.append("<w:pageBreakBefore/>")
    if num:
        p.append(f'<w:numPr><w:ilvl w:val="{num[1]}"/><w:numId w:val="{num[0]}"/></w:numPr>')
    if tabs:
        p.append("<w:tabs>" + "".join(f'<w:tab w:val="left" w:pos="{t}"/>' for t in tabs) + "</w:tabs>")
    sp = []
    if before is not None:
        sp.append(f'w:before="{before}"')
    if after is not None:
        sp.append(f'w:after="{after}"')
    if line is not None:
        sp.append(f'w:line="{line}" w:lineRule="auto"')
    if sp:
        p.append(f"<w:spacing {' '.join(sp)}/>")
    if ind:
        p.append("<w:ind " + " ".join(f'w:{k}="{v}"' for k, v in ind.items()) + "/>")
    if jc:
        p.append(f'<w:jc w:val="{jc}"/>')
    if mark_sz:
        p.append(f'<w:rPr><w:sz w:val="{mark_sz}"/><w:szCs w:val="{mark_sz}"/></w:rPr>')
    return f"<w:p><w:pPr>{''.join(p)}</w:pPr>{raw_runs}{runs_xml(doc, runs, size, bold)}</w:p>"


def text_of(runs: list[dict]) -> str:
    return "".join(r["t"] for r in runs)


# ------------------------------------------------------------------------------------------------ tables, figures

def scale_widths(widths: list[int] | None, n: int) -> list[int]:
    w = widths if widths and len(widths) == n else [1] * n
    tot = sum(w)
    out = [int(x * CONTENT_W / tot) for x in w]
    out[-1] += CONTENT_W - sum(out)
    return out


def table_xml(doc: Doc, b: dict) -> list[str]:
    n = len(b["header"])
    widths = scale_widths(b.get("widths"), n)
    first_col = [text_of([r for part in row[0] for r in part]) for row in b["rows"]]
    bold_first = bool(first_col) and max(len(x) for x in first_col) <= 30
    look_val = 0x0020 | 0x0400 | (0x0080 if bold_first else 0)
    tbl = ['<w:tbl><w:tblPr><w:tblStyle w:val="PlainTable1"/>'
           f'<w:tblW w:w="{CONTENT_W}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
           f'<w:tblLook w:val="{look_val:04X}" w:firstRow="1" w:lastRow="0" w:firstColumn="{int(bold_first)}" '
           'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"]

    def cell(parts: list[list[dict]], w: int, header: bool) -> str:
        ps = "".join(para(doc, part, keep_next=True, keep_lines=True, before=20, after=20, line=240,
                          size=TABLE_SZ, bold=header, mark_sz=TABLE_SZ) for part in parts)
        return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>{ps}</w:tc>'

    tbl.append('<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>'
               + "".join(cell(c, widths[i], True) for i, c in enumerate(b["header"])) + "</w:tr>")
    for row in b["rows"]:
        tbl.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>"
                   + "".join(cell(c, widths[i], False) for i, c in enumerate(row)) + "</w:tr>")
    tbl.append("</w:tbl>")
    out = ["".join(tbl)]
    notes = b.get("notes") or []
    # caption below the table, bold label, as in the template's example ("Table 1. An example of a simple table")
    out.append(para(doc, b["caption"], keep_next=bool(notes), keep_lines=True, before=120,
                    after=60 if notes else 360, jc="both"))
    for k, note in enumerate(notes):
        last = k == len(notes) - 1
        out.append(para(doc, note, keep_next=not last, keep_lines=True, before=0, after=360 if last else 40,
                        size=NOTE_SZ, mark_sz=NOTE_SZ, jc="both"))
    return out


def png_info(data: bytes) -> tuple[int, int, float | None]:
    """Width and height in pixels and the resolution in dots per inch (from the pHYs chunk, None without one)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "figures must be PNG"
    w, h = struct.unpack(">II", data[16:24])
    k = data.find(b"pHYs", 0, data.find(b"IDAT"))
    if k > 0:
        ppu, _, unit = struct.unpack(">IIB", data[k + 4:k + 13])
        if unit == 1:                                   # pixels per metre
            return w, h, ppu * 0.0254
    return w, h, None


def figure_xml(doc: Doc, b: dict, n: int) -> list[str]:
    data = Path(b["path"]).read_bytes()
    w, h, dpi = png_info(data)
    cx = min(FIG_W_EMU, round(w / dpi * 914400)) if dpi else FIG_W_EMU   # the size the figure was drawn for
    cy = int(cx * h / w)
    rid = doc.image(data)
    pid = doc.next_pic
    doc.next_pic += 1
    descr = escape(text_of(b["legend"])[:400], {'"': "&quot;"})
    drawing = (
        f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{pid}" name="Figure {n}" descr="{descr}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="{A}"><a:graphicData uri="{PIC}"><pic:pic xmlns:pic="{PIC}">'
        f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{Path(b["path"]).name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing></w:r>')
    return [para(doc, [], keep_next=True, before=240, after=0, jc="center", raw_runs=drawing),
            # legend below the figure, bold label, spacing as the template's "Figure 1." legend
            para(doc, b["legend"], keep_lines=True, before=187, after=360, jc="both")]


# ------------------------------------------------------------------------------------------------ body

def body_xml(doc: Doc, blocks: list[dict]) -> str:
    out: list[str] = []
    section = ""
    fig_n = 0
    prev = {"type": "start"}
    for k, b in enumerate(blocks):
        nxt = blocks[k + 1] if k + 1 < len(blocks) else {"type": "end"}
        t = b["type"]
        if t == "title":
            out.append(para(doc, [{"t": "Title"}], "Heading1", keep_next=True, before=120))
            out.append(para(doc, b["runs"], size=TITLE_SZ, bold=True, after=120, line=264))
        elif t == "h1":
            section = text_of(b["runs"])
            out.append(para(doc, b["runs"], "Heading1", keep_next=True, page_break=section in NEW_PAGE,
                            before=0 if section in NEW_PAGE else 240))
        elif t == "h2":
            out.append(para(doc, b["runs"], "Heading2", keep_next=True, before=200, after=60, ind={"left": 0}))
        elif t == "p":
            if section == "Authors and affiliations":
                is_aff = bool(b["runs"]) and b["runs"][0].get("sup") and b["runs"][0]["t"][:1].isdigit()
                nxt_aff = nxt["type"] == "p" and nxt.get("runs") and nxt["runs"][0].get("sup") \
                    and nxt["runs"][0]["t"][:1].isdigit()
                out.append(para(doc, b["runs"], after=0 if (is_aff and nxt_aff) else 160))
            elif section == "Keywords":
                out.append(para(doc, b["runs"], after=120))
            else:
                out.append(para(doc, b["runs"], after=160, line=256, jc="both"))
        elif t in ("bullet", "numbered"):
            last = nxt["type"] != t
            if t == "numbered" and prev["type"] != "numbered":
                doc.decimal_lists += 1                  # a new list, numbered from 1
            numid = BULLET_NUMID if t == "bullet" else DECIMAL_NUMID + doc.decimal_lists - 1
            out.append(para(doc, b["runs"], "ListParagraph", num=(numid, 0), tabs=[599], before=60,
                            after=160 if last else 0, line=256, size=BODY_SZ, mark_sz=BODY_SZ))
        elif t == "ref":
            n = b["n"].rstrip(".")
            out.append(para(doc, [{"t": f"{n}.\t"}] + b["runs"], tabs=[425], after=60,
                            ind={"left": 425, "hanging": 425}))
        elif t == "table":
            out += table_xml(doc, b)
        elif t == "figure":
            fig_n += 1
            out += figure_xml(doc, b, fig_n)
        else:
            raise ValueError(f"unknown block {t}")
        prev = b
    return "".join(out)


# ------------------------------------------------------------------------------------------------ package

DECIMAL_ABSTRACT = (
    '<w:abstractNum w:abstractNumId="3"><w:multiLevelType w:val="hybridMultilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
    '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="598" w:hanging="313"/></w:pPr>'
    '<w:rPr><w:rFonts w:ascii="Cambria" w:eastAsia="Cambria" w:hAnsi="Cambria" w:cs="Cambria" w:hint="default"/>'
    '<w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:lvl></w:abstractNum>')


def fill(blocks: list[dict], out_path: Path, title: str, authors: str) -> dict:
    zin = zipfile.ZipFile(TEMPLATE)
    parts = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    order = [i.filename for i in zin.infolist()]

    # document.xml: template prefix, the first two paragraphs, the manuscript, the template's sectPr
    d = parts["word/document.xml"].decode("utf-8")
    i0 = d.index("<w:body>") + len("<w:body>")
    j0 = d.rindex("<w:sectPr")
    paras = re.findall(r"<w:p[ >].*?</w:p>", d[i0:j0], re.S)
    head = paras[0] + paras[1]
    assert "Article Template" in re.sub(r"<[^>]+>", "", paras[0]), "unexpected template"
    doc = Doc()
    body = body_xml(doc, blocks)
    parts["word/document.xml"] = (d[:i0] + head + body + d[j0:]).encode("utf-8")

    # numbering: add a decimal list (abstractNum 3) and one instance of it per numbered list (num 4, 5, ...), the
    # later ones restarting at 1
    nx = parts["word/numbering.xml"].decode("utf-8")
    n_lists = max(1, doc.decimal_lists)
    assert 'w:abstractNumId="3"' not in nx
    assert not any(f'w:numId="{DECIMAL_NUMID + i}"' in nx for i in range(n_lists))
    k = nx.index("<w:num ")
    nx = nx[:k] + DECIMAL_ABSTRACT + nx[k:]
    nums = "".join(f'<w:num w:numId="{DECIMAL_NUMID + i}"><w:abstractNumId w:val="3"/>'
                   + ('<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride>' if i else "")
                   + "</w:num>" for i in range(n_lists))
    nx = nx.replace("</w:numbering>", nums + "</w:numbering>")
    parts["word/numbering.xml"] = nx.encode("utf-8")

    # relationships: drop those of the removed instruction text, add hyperlinks and figures
    rels = parts["word/_rels/document.xml.rels"].decode("utf-8")
    used = set(re.findall(r'r:(?:id|embed)="(rId\d+)"', parts["word/document.xml"].decode("utf-8")))
    kept, dropped = [], []
    for m in re.finditer(r"<Relationship [^>]*/>", rels):
        rel = m.group(0)
        rid = re.search(r'Id="([^"]+)"', rel).group(1)
        typ = re.search(r'Type="([^"]+)"', rel).group(1)
        if typ in (REL_HYPERLINK, REL_IMAGE) and rid not in used:
            dropped.append((rid, re.search(r'Target="([^"]+)"', rel).group(1)))
        else:
            kept.append(rel)
    for rid, typ, target, ext in doc.rels:
        kept.append(f'<Relationship Id="{rid}" Type="{typ}" Target="{escape(target, {chr(34): "&quot;"})}"'
                    + (' TargetMode="External"' if ext else "") + "/>")
    head_rels = rels[:rels.index("<Relationship ")]
    parts["word/_rels/document.xml.rels"] = (head_rels + "".join(kept) + "</Relationships>").encode("utf-8")
    for rid, target in dropped:
        if not target.startswith("http") and f"word/{target}" in parts:
            del parts[f"word/{target}"]
            order.remove(f"word/{target}")
    for name, data in doc.media:
        parts[name] = data
        order.append(name)

    # core properties: the article instead of the template
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = parts["docProps/core.xml"].decode("utf-8")
    core = re.sub(r"<dc:creator>.*?</dc:creator>", f"<dc:creator>{escape(authors)}</dc:creator>", core)
    core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>",
                  f"<cp:lastModifiedBy>{escape(authors.split(',')[0])}</cp:lastModifiedBy>", core)
    core = re.sub(r"<cp:revision>.*?</cp:revision>", "<cp:revision>1</cp:revision>", core)
    core = re.sub(r'(<dcterms:created xsi:type="dcterms:W3CDTF">).*?(</dcterms:created>)', rf"\g<1>{now}\g<2>", core)
    core = re.sub(r'(<dcterms:modified xsi:type="dcterms:W3CDTF">).*?(</dcterms:modified>)', rf"\g<1>{now}\g<2>", core)
    if "<dc:title>" not in core:
        core = core.replace("<dc:creator>", f"<dc:title>{escape(title)}</dc:title><dc:creator>", 1)
    parts["docProps/core.xml"] = core.encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (2026, 10, 16, 0, 0, 0)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(info, parts[name])
    return {"hyperlinks": len(doc.links), "figures": len(doc.media), "dropped_relationships": dropped,
            "unchanged_parts": sorted(n for n in order if n in zin.namelist() and parts[n] == zin.read(n))}


def main(argv: list[str]) -> int:
    src, out = Path(argv[1]), Path(argv[2])
    blocks = json.loads(src.read_text(encoding="utf-8"))["blocks"]
    title = text_of(next(b for b in blocks if b["type"] == "title")["runs"])
    authors = "Nikolaos D. Tantaroudas, Andrew J. McCracken, Ilias Karachalios, Evangelos Papatheou"
    info = fill(blocks, out, title, authors)
    print(f"ORE template: {out} ({out.stat().st_size} bytes); {info['figures']} figures, "
          f"{info['hyperlinks']} hyperlinks; template parts unchanged: {len(info['unchanged_parts'])}; "
          f"removed instruction relationships: {[r for r, _ in info['dropped_relationships']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
