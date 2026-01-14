"""
ELSST RDF Data Processing Script
================================
This script processes the ELSST (European Language Social Science Thesaurus) RDF data
to create positive and negative concept pairs for machine learning tasks.

Positive pairs: Concept prefLabel paired with its altLabel (entry terms)
Negative pairs: Concepts that are NOT related via:
    - Entry term relationship (altLabel)
    - Parent-child hierarchy (broader/narrower)
    - Ancestor-descendant relationship
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

RDF_FILE_PATH = "datasets/raw_datasets/ELSST_R5.rdf"
OUTPUT_DIR = "datasets/processed_datasets"
LANGUAGE = "en"  # Focus on English labels
RANDOM_SEED = 42

# Set to True to generate ALL negative pairs (can be millions!)
# Set to False to sample negatives to match positive pairs count
GENERATE_ALL_NEGATIVES = True

# ============================================================================
# RDF NAMESPACE DEFINITIONS
# ============================================================================

NAMESPACES = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
}


# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def parse_rdf_file(file_path):
    """
    Parse the RDF file and return the root element.
    """
    print(f"Parsing RDF file: {file_path}")
    tree = ET.parse(file_path)
    return tree.getroot()


def extract_concepts(root):
    """
    Extract all SKOS concepts from the RDF data.
    
    Returns:
        concepts: dict mapping concept URI to concept data
                  {uri: {'prefLabel': str, 'altLabels': list, 'broader': list, 'narrower': list}}
    """
    concepts = {}
    
    # Find all Description elements (which contain concepts)
    for description in root.findall('.//rdf:Description', NAMESPACES):
        uri = description.get(f'{{{NAMESPACES["rdf"]}}}about')
        
        if not uri:
            continue
            
        # Check if this is a SKOS Concept
        type_elem = description.find('rdf:type', NAMESPACES)
        if type_elem is None:
            continue
            
        type_resource = type_elem.get(f'{{{NAMESPACES["rdf"]}}}resource')
        if type_resource != 'http://www.w3.org/2004/02/skos/core#Concept':
            continue
        
        # Initialize concept data
        concept_data = {
            'prefLabel': None,
            'altLabels': [],
            'broader': [],
            'narrower': []
        }
        
        # Extract prefLabel (preferred label) for specified language
        for pref_label in description.findall('skos:prefLabel', NAMESPACES):
            lang = pref_label.get(f'{{{NAMESPACES["rdf"]}}}lang') or pref_label.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == LANGUAGE:
                concept_data['prefLabel'] = pref_label.text
                break
        
        # Extract altLabels (alternative labels / entry terms)
        for alt_label in description.findall('skos:altLabel', NAMESPACES):
            lang = alt_label.get(f'{{{NAMESPACES["rdf"]}}}lang') or alt_label.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == LANGUAGE and alt_label.text:
                concept_data['altLabels'].append(alt_label.text)
        
        # Extract broader concepts (parents)
        for broader in description.findall('skos:broader', NAMESPACES):
            broader_uri = broader.get(f'{{{NAMESPACES["rdf"]}}}resource')
            if broader_uri:
                concept_data['broader'].append(broader_uri)
        
        # Extract narrower concepts (children)
        for narrower in description.findall('skos:narrower', NAMESPACES):
            narrower_uri = narrower.get(f'{{{NAMESPACES["rdf"]}}}resource')
            if narrower_uri:
                concept_data['narrower'].append(narrower_uri)
        
        # Only add concepts that have a prefLabel in our target language
        if concept_data['prefLabel']:
            concepts[uri] = concept_data
    
    return concepts


# ============================================================================
# HIERARCHY PROCESSING FUNCTIONS
# ============================================================================

def get_all_ancestors(concept_uri, concepts, visited=None):
    """
    Recursively get all ancestors (parents, grandparents, etc.) of a concept.
    
    Args:
        concept_uri: URI of the concept
        concepts: dict of all concepts
        visited: set of already visited URIs (to prevent cycles)
    
    Returns:
        set of ancestor URIs
    """
    if visited is None:
        visited = set()
    
    ancestors = set()
    
    if concept_uri in visited:
        return ancestors
    
    visited.add(concept_uri)
    
    if concept_uri not in concepts:
        return ancestors
    
    # Get direct parents
    for parent_uri in concepts[concept_uri]['broader']:
        ancestors.add(parent_uri)
        # Recursively get ancestors of parents
        ancestors.update(get_all_ancestors(parent_uri, concepts, visited.copy()))
    
    return ancestors


def get_all_descendants(concept_uri, concepts, visited=None):
    """
    Recursively get all descendants (children, grandchildren, etc.) of a concept.
    
    Args:
        concept_uri: URI of the concept
        concepts: dict of all concepts
        visited: set of already visited URIs (to prevent cycles)
    
    Returns:
        set of descendant URIs
    """
    if visited is None:
        visited = set()
    
    descendants = set()
    
    if concept_uri in visited:
        return descendants
    
    visited.add(concept_uri)
    
    if concept_uri not in concepts:
        return descendants
    
    # Get direct children
    for child_uri in concepts[concept_uri]['narrower']:
        descendants.add(child_uri)
        # Recursively get descendants of children
        descendants.update(get_all_descendants(child_uri, concepts, visited.copy()))
    
    return descendants


def build_exclusion_sets(concepts):
    """
    Build exclusion sets for each concept.
    A concept cannot be paired with:
    - Itself
    - Its ancestors (parents, grandparents, etc.)
    - Its descendants (children, grandchildren, etc.)
    
    Returns:
        dict mapping concept URI to set of excluded URIs
    """
    print("Building exclusion sets for hierarchy relationships...")
    exclusion_sets = {}
    
    for uri in concepts:
        excluded = set()
        excluded.add(uri)  # Exclude self
        excluded.update(get_all_ancestors(uri, concepts))
        excluded.update(get_all_descendants(uri, concepts))
        exclusion_sets[uri] = excluded
    
    return exclusion_sets


# ============================================================================
# PAIR GENERATION FUNCTIONS
# ============================================================================

def generate_positive_pairs(concepts):
    """
    Generate positive pairs from entry terms (altLabels) of the same concept.
    
    For a concept with prefLabel P and altLabels [A1, A2, A3], we generate:
    - (A1, A2), (A1, A3), (A2, A3)        # all altLabel combinations
    
    NOTE: We do NOT pair prefLabel with altLabels because prefLabel IS the concept.
    A concept cannot be paired with itself.
    
    Returns:
        list of dicts with term pairs
    """
    print("Generating positive pairs (entry term combinations only)...")
    positive_pairs = []
    
    for uri, data in concepts.items():
        pref_label = data['prefLabel']
        alt_labels = data['altLabels']
        
        # Only generate pairs between altLabels (entry terms)
        # A concept (prefLabel) should NOT be paired with itself
        for i in range(len(alt_labels)):
            for j in range(i + 1, len(alt_labels)):
                positive_pairs.append({
                    'term1': alt_labels[i],
                    'term2': alt_labels[j],
                    'concept_uri': uri,
                    'concept_prefLabel': pref_label,
                    'label': 1  # Positive label
                })
    
    print(f"Generated {len(positive_pairs)} positive pairs")
    return positive_pairs


def generate_negative_pairs(concepts, exclusion_sets, sample_size=None):
    """
    Generate negative pairs between unrelated concepts.
    
    A negative pair is created between two concepts that:
    - Are NOT the same concept
    - Are NOT in a parent-child or ancestor-descendant relationship
    
    Args:
        concepts: dict of all concepts
        exclusion_sets: dict mapping concept URI to excluded URIs
        sample_size: if None, generate ALL pairs; otherwise sample this many
    
    Returns:
        list of dicts with term pairs
    """
    concept_uris = sorted(concepts.keys())
    total_concepts = len(concept_uris)
    
    # First, count total possible negative pairs (for statistics)
    print("Counting possible negative pairs...")
    total_possible = 0
    valid_pairs_indices = []
    
    for i in range(total_concepts):
        uri1 = concept_uris[i]
        excluded1 = exclusion_sets[uri1]
        
        for j in range(i + 1, total_concepts):
            uri2 = concept_uris[j]
            if uri2 not in excluded1:
                valid_pairs_indices.append((i, j))
                total_possible += 1
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1}/{total_concepts} concepts...")
    
    print(f"Total valid negative pairs possible: {total_possible}")
    
    # Decide whether to generate all or sample
    if sample_size is not None and sample_size < total_possible:
        print(f"Sampling {sample_size} negative pairs...")
        random.seed(RANDOM_SEED)
        sampled_indices = random.sample(valid_pairs_indices, sample_size)
        pairs_to_generate = sampled_indices
    else:
        print(f"Generating ALL {total_possible} negative pairs...")
        pairs_to_generate = valid_pairs_indices
    
    # Generate the pairs
    negative_pairs = []
    for idx, (i, j) in enumerate(pairs_to_generate):
        uri1 = concept_uris[i]
        uri2 = concept_uris[j]
        label1 = concepts[uri1]['prefLabel']
        label2 = concepts[uri2]['prefLabel']
        
        negative_pairs.append({
            'term1': label1,
            'term2': label2,
            'concept1_uri': uri1,
            'concept2_uri': uri2,
            'label': 0
        })
        
        # Progress indicator for large generations
        if (idx + 1) % 100000 == 0:
            print(f"  Generated {idx + 1}/{len(pairs_to_generate)} pairs...")
    
    print(f"Generated {len(negative_pairs)} negative pairs")
    return negative_pairs, total_possible


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_pairs_to_json(pairs, filename):
    """Save pairs to a JSON file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {filepath}")


def save_combined_dataset(positive_pairs, negative_pairs, filename):
    """
    Combine positive and negative pairs into a single dataset.
    """
    combined = positive_pairs + negative_pairs
    random.seed(RANDOM_SEED)
    random.shuffle(combined)
    save_pairs_to_json(combined, filename)
    return combined


def print_statistics(concepts, positive_pairs, negative_pairs, exclusion_sets, total_possible_negatives):
    """Print detailed dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS REPORT")
    print("=" * 70)
    
    # Concept statistics
    concepts_with_alt = sum(1 for c in concepts.values() if c['altLabels'])
    concepts_with_multiple_alt = sum(1 for c in concepts.values() if len(c['altLabels']) >= 2)
    concepts_without_alt = len(concepts) - concepts_with_alt
    total_alt_labels = sum(len(c['altLabels']) for c in concepts.values())
    
    print(f"\n--- CONCEPT STATISTICS ---")
    print(f"Total concepts extracted:          {len(concepts)}")
    print(f"Concepts WITH altLabels:           {concepts_with_alt}")
    print(f"Concepts with 2+ altLabels:        {concepts_with_multiple_alt}")
    print(f"Concepts WITHOUT altLabels:        {concepts_without_alt}")
    print(f"Total altLabels (entry terms):     {total_alt_labels}")
    
    # Positive pairs breakdown
    print(f"\n--- POSITIVE PAIRS (entry terms of same concept) ---")
    print(f"Total positive pairs:              {len(positive_pairs)}")
    
    # Calculate: only altLabel-altLabel pairs (no prefLabel involved)
    alt_alt_pairs = 0
    for uri, data in concepts.items():
        n_alt = len(data['altLabels'])
        alt_alt_pairs += (n_alt * (n_alt - 1)) // 2  # combinations of altLabels
    
    print(f"  (Only altLabel ↔ altLabel pairs)")
    print(f"  Note: Concept (prefLabel) is NOT paired with entry terms")
    
    # Negative pairs statistics
    print(f"\n--- NEGATIVE PAIRS (unrelated concepts) ---")
    print(f"Generated negative pairs:          {len(negative_pairs)}")
    print(f"Total possible negative pairs:     {total_possible_negatives}")
    
    # Calculate theoretical maximum and excluded pairs
    max_possible = (len(concepts) * (len(concepts) - 1)) // 2
    excluded_pairs = max_possible - total_possible_negatives
    print(f"Max concept pair combinations:     {max_possible}")
    print(f"Excluded (hierarchical relations): {excluded_pairs}")
    
    # Total
    print(f"\n--- FINAL DATASET ---")
    print(f"Total positive pairs:            {len(positive_pairs)}")
    print(f"Total negative pairs:            {len(negative_pairs)}")
    print(f"Grand total pairs:               {len(positive_pairs) + len(negative_pairs)}")
    print("=" * 70)


def print_sample_pairs(positive_pairs, negative_pairs, num_samples=5):
    """Print sample pairs for verification."""
    print("\n" + "=" * 70)
    print("SAMPLE POSITIVE PAIRS (entry terms of same concept)")
    print("=" * 70)
    for pair in positive_pairs[:num_samples]:
        print(f"  Term 1:  {pair['term1']}")
        print(f"  Term 2:  {pair['term2']}")
        print(f"  Concept: {pair['concept_prefLabel']}")
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("SAMPLE NEGATIVE PAIRS (unrelated concepts)")
    print("=" * 70)
    for pair in negative_pairs[:num_samples]:
        print(f"  Term 1: {pair['term1']}")
        print(f"  Term 2: {pair['term2']}")
        print("-" * 50)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("ELSST RDF DATA PROCESSING")
    print("=" * 70 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Parse the RDF file
    root = parse_rdf_file(RDF_FILE_PATH)
    
    # Step 2: Extract concepts
    concepts = extract_concepts(root)
    print(f"Extracted {len(concepts)} concepts with English labels")
    
    # Step 3: Build exclusion sets for hierarchy
    exclusion_sets = build_exclusion_sets(concepts)
    
    # Step 4: Generate positive pairs (ALL combinations within same concept)
    positive_pairs = generate_positive_pairs(concepts)
    
    # Step 5: Generate negative pairs
    if GENERATE_ALL_NEGATIVES:
        # Generate ALL possible negative pairs
        negative_pairs, total_possible_negatives = generate_negative_pairs(
            concepts, exclusion_sets, sample_size=None
        )
    else:
        # Sample negatives to match positive pairs count
        negative_pairs, total_possible_negatives = generate_negative_pairs(
            concepts, exclusion_sets, sample_size=len(positive_pairs)
        )
    
    # Step 6: Print statistics and samples
    print_statistics(concepts, positive_pairs, negative_pairs, exclusion_sets, total_possible_negatives)
    print_sample_pairs(positive_pairs, negative_pairs)
    
    # Step 7: Save datasets
    print("\nSaving datasets...")
    save_pairs_to_json(positive_pairs, "elsst_positive_pairs.json")
    save_pairs_to_json(negative_pairs, "elsst_negative_pairs.json")
    combined = save_combined_dataset(positive_pairs, negative_pairs, "elsst_combined_pairs.json")
    
    # Step 8: Save concept hierarchy for reference
    concept_summary = {
        uri: {
            'prefLabel': data['prefLabel'],
            'altLabels': data['altLabels'],
            'num_ancestors': len(exclusion_sets[uri]) - 1 - len(data['narrower']),
            'num_descendants': len(get_all_descendants(uri, concepts))
        }
        for uri, data in concepts.items()
    }
    save_pairs_to_json(concept_summary, "elsst_concepts_summary.json")
    
    print("\nProcessing complete!")
    return concepts, positive_pairs, negative_pairs


if __name__ == "__main__":
    main()
