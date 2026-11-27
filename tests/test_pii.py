from server.entity_linker import EntityLinker, entity_hashes, normalize
from server.pii_detector import PIIDetector, cnp_control_digit, cnp_is_valid, entity_boxes, iban_is_valid
from server.vault import Entity, Vault, Word


def make_words(lines: list[str]) -> list[Word]:
    words = []
    for li, line in enumerate(lines):
        x = 10
        for tok in line.split():
            words.append(Word(tok, x, 10 + li * 40, 12 * len(tok), 30, 95.0, 1, (1, 1, 1, li)))
            x += 12 * len(tok) + 12
    return words


def test_cnp_checksum():
    first12 = "196052333123"
    good = first12 + cnp_control_digit(first12)
    assert cnp_is_valid(good)
    bad = first12 + str((int(good[-1]) + 1) % 10)
    assert not cnp_is_valid(bad)
    assert not cnp_is_valid("1961332331234")   # month 13
    assert not cnp_is_valid("0960523331234")   # sex digit 0


def test_iban():
    assert iban_is_valid("RO49AAAA1B31007593840000")
    assert not iban_is_valid("RO49AAAA1B31007593840001")


def test_detector_finds_core_entities():
    first12 = "296101233123"
    cnp = first12 + cnp_control_digit(first12)
    words = make_words([
        "CERERE",
        "Subsemnata Ionescu Maria-Elena, CNP " + cnp,
        "domiciliata in Str. Florilor nr. 12, Suceava",
        "posesoare a CI SERIA SV NR 123456 tel 0745 123 456",
        "Semnatura: Ionescu",
    ])
    ents = PIIDetector(use_spacy=False).detect(words)
    types = {e.type for e in ents}
    assert {"CNP", "PERSON_NAME", "ADDRESS", "ID_SERIES", "PHONE"} <= types
    assert any(e.type == "CNP" and e.value == cnp for e in ents)
    assert any(e.type == "ID_SERIES" and e.value == "SV123456" for e in ents)
    # signature occurrence is propagated
    last_line_words = {i for i, w in enumerate(words) if w.line_key[3] == 4}
    assert any(e.type == "PERSON_NAME" and set(e.word_indices) & last_line_words for e in ents)
    assert entity_boxes(ents, words)


def test_name_normalization_is_order_and_diacritic_insensitive():
    assert normalize("PERSON_NAME", "Ștefănescu Ana-Maria") == normalize("PERSON_NAME", "ANA MARIA STEFANESCU")


def test_linker_clusters_by_shared_hash():
    v = Vault(secret=b"x" * 32)
    linker = EntityLinker()
    cnp = "296101233123" + cnp_control_digit("296101233123")
    linker.update_document("d1", entity_hashes([Entity("CNP", cnp, [0], 0.99)], v.hmac_hex))
    linker.update_document("d2", entity_hashes([Entity("CNP", cnp, [0], 0.99),
                                                Entity("PERSON_NAME", "Ana Pop", [1], 0.9)], v.hmac_hex))
    linker.update_document("d3", entity_hashes([Entity("PERSON_NAME", "Ion Vasile", [0], 0.9)], v.hmac_hex))
    res = linker.link()
    assert len(res["clusters"]) == 1
    assert res["clusters"][0]["doc_ids"] == ["d1", "d2"]
    assert res["clusters"][0]["shared_entity_types"] == ["CNP"]
    assert res["unlinked_doc_ids"] == ["d3"]
    assert cnp not in str(res)


def test_unlabeled_full_names_and_ocr_capital_i():
    words = make_words([
        "Ionescu Maria Elena a depus cererea.",
        "NICHIFOR ANTONIU",
        "Subsemnata lonescu Maria, CNP lipsa",          # OCR read the capital I as l
        "Furnizor: SC Energie Verde SRL",
        "domiciliat in Str. Florilor nr. 3, Mun. Iasi",
        "Domnule Director, va rog",
    ])
    ents = PIIDetector(use_spacy=False).detect(words)
    names = {e.value for e in ents if e.type == "PERSON_NAME"}
    assert "Ionescu Maria Elena" in names
    assert "NICHIFOR ANTONIU" in names
    assert any(v.startswith("Ionescu Maria") for v in names if v != "Ionescu Maria Elena")   # 'l' fixed to 'I'
    flat = " ".join(names)
    assert "Energie" not in flat and "Florilor" not in flat and "Director" not in flat and "Iasi" not in flat


def test_romanian_id_card_fields():
    first12 = "182031335597"
    cnp = first12 + cnp_control_digit(first12)
    words = make_words([
        "ROMANIA ROUMANIE ROMANIA",
        "CARTE DE IDENTITATE / IDENTITY CARD",
        f"SERIA TM NR 632084 CNP {cnp}",
        "Nume/Nom/Last name",
        "DINU",
        "Prenume/Prenom/First name",
        "NICHIFOR ANTONIU",
        "Loc nastere/Lieu de naissance/Place of birth",
        "Jud. TM Mun. Timisoara",
        "Domiciliu/Adresse/Address",
        "Str. Lalelelor nr. 10, bl. E2, ap. 24",
        "Emisa de/Delivree par/Issued by SPCLEP Timis",
        "Valabilitate/Validite/Validity 20.05.2023-20.05.2033",
        "IDROUDINU<<NICHIFOR<ANTONIU<<<<<<<<<",
        f"TM632084<4ROU820313M3205238{cnp[7:]}<<<",
    ])
    ents = PIIDetector(use_spacy=False).detect(words)
    by_type: dict[str, list[str]] = {}
    for e in ents:
        by_type.setdefault(e.type, []).append(e.value)
    assert cnp in by_type["CNP"]
    assert "TM632084" in by_type["ID_SERIES"]
    assert "DINU" in by_type["PERSON_NAME"] and "NICHIFOR ANTONIU" in by_type["PERSON_NAME"]
    assert any("Lalelelor" in v for v in by_type["ADDRESS"])
    assert any("SPCLEP Timis" in v for v in by_type["ISSUER"])
    assert "20.05.2023-20.05.2033" in by_type["ID_DATE"]
    assert any("Timisoara" in v for v in by_type["BIRTH_PLACE"])
    assert len(by_type["MRZ"]) == 2


def test_issuer_and_validity_only_on_id_documents():
    words = make_words(["FACTURA FISCALA Nr. EV-1234 Data emiterii: 05.04.2024",
                        "Valabilitate oferta 01.01.2024-31.01.2024"])
    types = {e.type for e in PIIDetector(use_spacy=False).detect(words)}
    assert "ID_DATE" not in types and "ISSUER" not in types


def test_portrait_box_on_id_card_layout():
    from server.pii_detector import portrait_boxes
    from server.vault import Page
    lines = ["CARTE DE IDENTITATE", "Nume/Nom/Last name", "DINU", "Prenume/Prenom/First name", "ANA",
             "Domiciliu/Adresse/Address", "Str. Lalelelor nr. 10", "IDROUDINU<<ANA<<<<<<<<<<<<<<"]
    words = []
    for li, line in enumerate(lines):
        x = 20 if li in (0, len(lines) - 1) else 520           # labels start right of the photo
        for tok in line.split():
            words.append(Word(tok, x, 20 + li * 60, 14 * len(tok), 30, 95.0, 1, (1, 1, 1, li)))
            x += 14 * len(tok) + 14
    page = Page(1, "missing.png", 1500, 520)
    boxes = portrait_boxes(words, [page])
    assert len(boxes) == 1
    _, x, y, w, h = boxes[0]
    assert x < 100 and x + w < 520 and h > 200
    # a plain form (text starts at the left margin) gets no portrait box
    assert portrait_boxes(make_words(["CARTE DE IDENTITATE seria", "Subsemnatul Pop Ion"]), [page]) == []


def test_redacted_id_card_labels_alone_are_not_pii():
    """After redaction only the field labels remain; a label must never be read as the next field's value."""
    words = make_words([
        "ROMANIA CARTE DE IDENTITATE / IDENTITY CARD",
        "SERIA CNP",
        "Nume/Nom/Last name",
        "Prenume/Prenom/First name",
        "Loc nastere/Lieu de naissance/Place of birth",
        "Domiciliu/Adresse/Address",
        "Emisa de/Delivree par/Issued by",
        "Valabilitate/Validite/Validity",
        "Cetatenie/Nationalite/Nationality",
        "Romana / ROU",
    ])
    assert PIIDetector(use_spacy=False).detect(words) == []


def test_electronic_id_card_2021_layout_with_ocr_junk():
    """New CEI front: multi-column labels, values on the next line, spaced dates, 2+7 document number,
    a specimen CNP with a wrong checksum, and the OCR junk seen on a real scan."""
    words = make_words([
        "| | | | [ROMANIA __CARTEDE",
        "| IDENTITY CARD | |",
        "| Nume /Surmame | AR |",
        ">>: @) MANOLE A",
        "N Prenume /G/ven names",
        "| CORINA",
        "IOANA",
        ", 4 Sex / Sex Cetatenie Data nasterii {Date of birth",
        "| F ROU 03 07 1983",
        "| | CNP PIN",
        r" j 2830703460094",
        "| . Nr. document / Document no. Data expirarii / Date of expiry",
        "—- 1 (4)SP1236343 (5) 06 04 2035",
        ". ( Semnatura titularului / Hofder's 6 )Q 1 3 [A 7 9",
    ])
    by_type: dict[str, list[str]] = {}
    for e in PIIDetector(use_spacy=False).detect(words):
        by_type.setdefault(e.type, []).append(e.value)
    assert "MANOLE" in by_type["PERSON_NAME"] and "CORINA IOANA" in by_type["PERSON_NAME"]
    assert "2830703460094" in by_type["CNP"]                  # redacted even though the checksum fails
    assert "SP1236343" in by_type["ID_SERIES"]
    assert "03 07 1983" in by_type["BIRTH_DATE"]
    assert "06 04 2035" in by_type["ID_DATE"]
    assert by_type.get("CAN")                                  # card access number next to the signature label
    assert "ROU" not in " ".join(v for vs in by_type.values() for v in vs)
