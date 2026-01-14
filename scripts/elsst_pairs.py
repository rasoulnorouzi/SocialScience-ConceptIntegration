#!/usr/bin/env python3
"""
Build positive and negative concept-entry term pairs from ELSST RDF.
"""

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}
SKOS_CONCEPT = "http://www.w3.org/2004/02/skos/core#Concept"
LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"


def parse_concepts(rdf_path, lang):
    concept_labels = defaultdict(set)
    entry_terms = defaultdict(set)
    parents = defaultdict(set)

    for _, elem in ET.iterparse(rdf_path, events=("end",)):
        if elem.tag != f"{{{NS['rdf']}}}Description":
            continue

        types = {
            t.get(f"{{{NS['rdf']}}}resource") for t in elem.findall("rdf:type", NS)
        }
        if SKOS_CONCEPT not in types:
            elem.clear()
            continue

        concept_id = elem.get(f"{{{NS['rdf']}}}about")
        if not concept_id:
            elem.clear()
            continue

        for pref in elem.findall("skos:prefLabel", NS):
            if lang and pref.get(LANG_ATTR) != lang:
                continue
            text = (pref.text or "").strip()
            if text:
                concept_labels[concept_id].add(text)
                entry_terms[concept_id].add(text)

        for alt in elem.findall("skos:altLabel", NS):
            if lang and alt.get(LANG_ATTR) != lang:
                continue
            text = (alt.text or "").strip()
            if text:
                entry_terms[concept_id].add(text)

        for broader in elem.findall("skos:broader", NS):
            parent = broader.get(f"{{{NS['rdf']}}}resource")
            if parent:
                parents[concept_id].add(parent)

        for narrower in elem.findall("skos:narrower", NS):
            child = narrower.get(f"{{{NS['rdf']}}}resource")
            if child:
                parents[child].add(concept_id)

        elem.clear()

    concept_ids = set(concept_labels) | set(entry_terms)
    parents = {
        cid: {p for p in plist if p in concept_ids} for cid, plist in parents.items()
    }
    return concept_labels, entry_terms, parents, concept_ids


def compute_ancestors(parents, concept_ids):
    memo = {}
    visiting = set()

    def dfs(concept_id):
        if concept_id in memo:
            return memo[concept_id]
        if concept_id in visiting:
            return set()

        visiting.add(concept_id)
        result = set()
        for parent in parents.get(concept_id, set()):
            result.add(parent)
            result.update(dfs(parent))
        visiting.remove(concept_id)

        memo[concept_id] = result
        return result

    for cid in concept_ids:
        dfs(cid)

    return memo


def write_pairs(output_dir, concept_labels, entry_terms, ancestors):
    os.makedirs(output_dir, exist_ok=True)
    pos_path = os.path.join(output_dir, "elsst_positive_pairs.csv")
    neg_path = os.path.join(output_dir, "elsst_negative_pairs.csv")

    all_terms = set()
    for terms in entry_terms.values():
        all_terms.update(terms)
    all_terms_sorted = sorted(all_terms)

    pos_count = 0
    neg_count = 0

    with open(pos_path, "w", newline="", encoding="utf-8") as pos_f, open(
        neg_path, "w", newline="", encoding="utf-8"
    ) as neg_f:
        pos_writer = csv.writer(pos_f)
        neg_writer = csv.writer(neg_f)
        header = ["concept_id", "concept_label", "entry_term"]
        pos_writer.writerow(header)
        neg_writer.writerow(header)

        for concept_id in sorted(concept_labels):
            labels = sorted(concept_labels[concept_id])
            if not labels:
                continue

            own_terms = entry_terms.get(concept_id, set())
            own_terms_sorted = sorted(own_terms)

            banned_terms = set(own_terms)
            for ancestor in ancestors.get(concept_id, set()):
                banned_terms.update(entry_terms.get(ancestor, set()))

            for label in labels:
                for term in own_terms_sorted:
                    pos_writer.writerow([concept_id, label, term])
                    pos_count += 1

                for term in all_terms_sorted:
                    if term in banned_terms:
                        continue
                    neg_writer.writerow([concept_id, label, term])
                    neg_count += 1

    return pos_path, neg_path, pos_count, neg_count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate positive/negative pairs from ELSST RDF with ancestor-aware "
            "negative filtering."
        )
    )
    parser.add_argument(
        "--rdf",
        default="datasets/raw_datasets/ELSST_R5.rdf",
        help="Path to ELSST RDF file.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code (e.g., en). Use 'all' to keep every language.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/processed",
        help="Directory to write the CSV outputs.",
    )
    args = parser.parse_args()

    lang = args.lang.strip()
    if lang.lower() in {"all", "*"}:
        lang = None

    concept_labels, entry_terms, parents, concept_ids = parse_concepts(
        args.rdf, lang
    )
    ancestors = compute_ancestors(parents, concept_ids)
    pos_path, neg_path, pos_count, neg_count = write_pairs(
        args.output_dir, concept_labels, entry_terms, ancestors
    )

    print(f"Wrote {pos_count} positive pairs to {pos_path}")
    print(f"Wrote {neg_count} negative pairs to {neg_path}")


if __name__ == "__main__":
    main()
