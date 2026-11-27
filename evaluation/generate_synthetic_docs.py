"""Generate 12 synthetic (FICTIONAL) Romanian documents with ground truth.

4 fictional people x 3 documents (identity card, application form, utility invoice), rendered
with different fonts, rotations, Gaussian noise and blur. Persons 3 and 4 deliberately share a
surname (hard negative for linking). Every page carries a "SPECIMEN - DATE FICTIVE" header.

Ground truth (synthetic_data/ground_truth.json) lists every PII occurrence (type + value)
and the person each document belongs to.

Usage: python -m evaluation.generate_synthetic_docs [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from server.pii_detector import cnp_control_digit

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "synthetic_data"
FONTS = {
    "arial": "arial.ttf", "times": "times.ttf", "courier": "cour.ttf", "calibri": "calibri.ttf",
    "verdana": "verdana.ttf", "georgia": "georgia.ttf", "tahoma": "tahoma.ttf",
}
MRZ_FONT = "cour.ttf"
COUNTIES = [("Suceava", "SV", 33), ("Iasi", "IS", 22), ("Cluj", "CJ", 12), ("Brasov", "BV", 8),
            ("Bacau", "BC", 4), ("Timis", "TM", 35)]
STREETS = ["Lalelelor", "Florilor", "Mihai Eminescu", "Stefan cel Mare", "Unirii", "Zorilor", "Teilor",
           "Castanilor", "Primaverii", "Independentei"]
BANKS = ["BTRL", "RNCB", "BRDE", "INGB", "RZBR", "CECE"]


def ascii_fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@dataclass
class Person:
    pid: str
    surname: str
    given: str
    sex: str
    cnp: str
    birth: str
    series: str
    number: str
    street: str
    city: str
    county: str
    county_code: str
    phone: str
    email: str
    iban: str

    @property
    def full(self) -> str:
        return f"{self.surname} {self.given}"

    @property
    def address_l1(self) -> str:
        return self.street

    @property
    def address_l2(self) -> str:
        return f"Mun. {self.city}, jud. {self.county}"


def make_cnp(rng: random.Random, sex: str, year: int, month: int, day: int, county_no: int) -> str:
    s = {"M": "1", "F": "2"}[sex] if year < 2000 else {"M": "5", "F": "6"}[sex]
    first12 = f"{s}{year % 100:02d}{month:02d}{day:02d}{county_no:02d}{rng.randint(1, 999):03d}"
    return first12 + cnp_control_digit(first12)


def make_iban(rng: random.Random) -> str:
    bban = rng.choice(BANKS) + "".join(rng.choice("0123456789") for _ in range(16))
    rearranged = bban + "RO00"
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    check = 98 - int(digits) % 97
    return f"RO{check:02d}{bban}"


def make_people(rng: random.Random, fake: Faker) -> list[Person]:
    people = []
    shared_surname = fake.last_name()
    for i in range(4):
        sex = "M" if i % 2 == 0 else "F"
        surname = shared_surname if i >= 2 else fake.last_name()
        given = (fake.first_name_male() if sex == "M" else fake.first_name_female())
        if i in (0, 3):
            given += " " + (fake.first_name_male() if sex == "M" else fake.first_name_female())
        year, month, day = rng.randint(1962, 2004), rng.randint(1, 12), rng.randint(1, 28)
        county, code, cno = rng.choice(COUNTIES)
        street = f"Str. {rng.choice(STREETS)} nr. {rng.randint(1, 120)}, bl. {rng.choice('ABCDE')}{rng.randint(1, 9)}, ap. {rng.randint(1, 60)}"
        folded = ascii_fold(f"{given.split()[0]}.{surname}").lower().replace(" ", "")
        people.append(Person(
            pid=f"P{i + 1}", surname=surname, given=given, sex=sex,
            cnp=make_cnp(rng, sex, year, month, day, cno), birth=f"{day:02d}.{month:02d}.{year}",
            series=code if rng.random() < 0.5 else rng.choice(["RX", "RT", "XV", "KX"]),
            number=f"{rng.randint(100000, 999999)}", street=street, city=county, county=county,
            county_code=code, phone=f"07{rng.randint(20, 89)} {rng.randint(100, 999)} {rng.randint(100, 999)}",
            email=f"{folded}{rng.randint(1, 99)}@example.com", iban=make_iban(rng)))
    return people


# ----------------------------------------------------------------------------- rendering
Segment = tuple[str, str | None]          # (text, pii_type or None)


def render(lines: list[list[Segment]], font_name: str, size: int, width: int, line_gap: float,
           mrz_from: int | None = None, border: bool = False) -> tuple[Image.Image, list[dict]]:
    font = ImageFont.truetype(FONTS[font_name], size)
    mrz_font = ImageFont.truetype(MRZ_FONT, size)
    height = int(80 + len(lines) * size * line_gap + 60)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    if border:
        d.rounded_rectangle((15, 15, width - 15, height - 15), radius=30, outline=(70, 90, 140), width=4)
    gt: list[dict] = []
    y = 50
    for li, segs in enumerate(lines):
        f = mrz_font if mrz_from is not None and li >= mrz_from else font
        x = 60
        for text, ptype in segs:
            if ptype:
                bbox = d.textbbox((x, y), text, font=f)
                gt.append({"type": ptype, "value": text, "bbox": list(bbox)})
            d.text((x, y), text, fill=(15, 15, 15), font=f)
            x = d.textbbox((x, y), text, font=f)[2]
        y += int(size * line_gap)
    return img, gt


def degrade(img: Image.Image, rng: random.Random, angle: float, noise: float, blur: float) -> Image.Image:
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img).astype(np.float32)
    if noise:
        nprng = np.random.default_rng(rng.randint(0, 10 ** 6))
        arr = np.clip(arr + nprng.normal(0, noise, arr.shape), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


HEADER: list[Segment] = [("SPECIMEN - DATE FICTIVE / SYNTHETIC TEST DATA", None)]


def mrz_lines(p: Person) -> list[str]:
    sur = ascii_fold(p.surname).upper().replace(" ", "<").replace("-", "<")
    giv = ascii_fold(p.given).upper().replace(" ", "<").replace("-", "<")
    l1 = f"IDROU{sur}<<{giv}".ljust(36, "<")[:36]
    l2 = f"{p.series}{p.number}<4ROU{p.cnp[1:7]}{'M' if p.sex == 'M' else 'F'}3205238{p.cnp[7:13]}".ljust(36, "<")[:36]
    return [l1, l2]


def id_card(p: Person) -> list[list[Segment]]:
    m1, m2 = mrz_lines(p)
    return [
        HEADER,
        [("ROMANIA  ROUMANIE  ROMANIA", None)],
        [("CARTE DE IDENTITATE / IDENTITY CARD", None)],
        [("SERIA ", None), (f"{p.series} NR {p.number}", "ID_SERIES"), ("    CNP ", None), (p.cnp, "CNP")],
        [("Nume/Nom/Last name", None)],
        [(p.surname.upper(), "PERSON_NAME")],
        [("Prenume/Prenom/First name", None)],
        [(p.given.upper(), "PERSON_NAME")],
        [("Cetatenie/Nationalite/Nationality", None)],
        [("Romana / ROU", None)],
        [("Domiciliu/Adresse/Address", None)],
        [(p.address_l1, "ADDRESS")],
        [(p.address_l2, "ADDRESS")],
        [(f"Emisa de/Delivree par/Issued by  SPCLEP {p.city}", None)],
        [("Valabilitate/Validite/Validity  20.05.2023-20.05.2033", None)],
        [(m1, "MRZ")],
        [(m2, "MRZ")],
    ]


def form(p: Person) -> list[list[Segment]]:
    sub = "Subsemnatul" if p.sex == "M" else "Subsemnata"
    born = "nascut" if p.sex == "M" else "nascuta"
    dom = "domiciliat" if p.sex == "M" else "domiciliata"
    return [
        HEADER,
        [("CERERE PENTRU ELIBERAREA UNEI ADEVERINTE", None)],
        [("Domnule Director,", None)],
        [(f"{sub} ", None), (p.full, "PERSON_NAME"), (f", {born} la data de ", None), (p.birth, "BIRTH_DATE"), (",", None)],
        [("CNP ", None), (p.cnp, "CNP"), (", ", None), (f"{dom} in ", None), (p.address_l1, "ADDRESS")],
        [(p.address_l2, "ADDRESS"), (",", None)],
        [("posesor al CI SERIA ", None), (f"{p.series} NR {p.number}", "ID_SERIES"), (", telefon ", None), (p.phone, "PHONE")],
        [("e-mail ", None), (p.email, "EMAIL"), (",", None)],
        [("va rog sa-mi eliberati o adeverinta de venit necesara la banca.", None)],
        [("Data: 14.03.2024", None)],
        [("Semnatura: ", None), (p.surname, "PERSON_NAME")],
    ]


def invoice(p: Person, rng: random.Random) -> list[list[Segment]]:
    kwh = rng.randint(120, 400)
    total = round(kwh * 0.92, 2)
    return [
        HEADER,
        [(f"FACTURA FISCALA Nr. EV-{rng.randint(10000, 99999)}   Data emiterii: 05.04.2024", None)],
        [("Furnizor: SC ENERGIE VERDE SRL   CUI 40517789", None)],
        [("Client: ", None), (p.full, "PERSON_NAME")],
        [("Adresa: ", None), (f"{p.address_l1}, {p.city}", "ADDRESS")],
        [("Telefon: ", None), (p.phone, "PHONE"), ("   Email: ", None), (p.email, "EMAIL")],
        [("IBAN client: ", None), (" ".join(p.iban[i:i + 4] for i in range(0, len(p.iban), 4)), "IBAN")],
        [(f"Energie electrica activa   {kwh} kWh   {total:.2f} lei", None)],
        [(f"Total de plata: {total * 1.19:.2f} lei", None)],
        [("Semnatura client: ", None), (p.surname, "PERSON_NAME")],
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT), help="Output folder (default: synthetic_data)")
    a = ap.parse_args(argv)
    seed = a.seed
    out = Path(a.out).resolve()
    rng = random.Random(seed)
    fake = Faker("ro_RO")
    Faker.seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    people = make_people(rng, fake)

    font_cycle = ["arial", "times", "courier", "calibri", "verdana", "georgia", "tahoma"]
    angles = [0.0, 1.5, -2.0, 3.0, -1.0, 2.5, -3.5, 0.8, -2.8, 1.2, 4.0, -0.6]
    docs = []
    k = 0
    for p in people:
        for kind in ("id_card", "form", "invoice"):
            lines = id_card(p) if kind == "id_card" else form(p) if kind == "form" else invoice(p, rng)
            font = font_cycle[k % len(font_cycle)]
            img, gt = render(lines, font, size=34 if kind != "id_card" else 32, width=1750 if kind == "form" else 1500,
                             line_gap=1.75, mrz_from=len(lines) - 2 if kind == "id_card" else None,
                             border=kind == "id_card")
            img = degrade(img, rng, angle=angles[k], noise=[0, 6, 12][k % 3], blur=[0, 0.5, 0.8][(k // 3) % 3])
            ext = "jpg" if kind == "invoice" else "png"
            name = f"{k + 1:02d}_{kind}_{p.pid}.{ext}"
            if ext == "jpg":
                img.save(out / name, quality=85)
            else:
                img.save(out / name)
            docs.append({"file": (out / name).relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out / name), "person": p.pid, "kind": kind, "font": font,
                         "rotation_deg": angles[k], "entities": [{"type": g["type"], "value": g["value"]} for g in gt]})
            k += 1
    (out / "ground_truth.json").write_text(json.dumps({"seed": seed, "note": "All data is synthetic/fictional.",
                                                       "documents": docs}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"Generated {len(docs)} documents in {out}")


if __name__ == "__main__":
    main()
