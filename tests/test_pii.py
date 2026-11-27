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
