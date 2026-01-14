"""
ELSST RDF Data Processing - Train/Test Split
=============================================
This script processes the ELSST thesaurus data to create train and test datasets
with positive and negative concept pairs.

- Train set: 70% of concepts
- Test set: 30% of concepts (completely different from train)
- Positive pairs: Entry terms (altLabels) of the same concept
- Negative pairs: Unrelated concepts (no hierarchical relationship)

IMPORTANT RULES:
1. A concept cannot pair with itself
2. Concepts with parent/ancestor relationship cannot be negative pairs
3. Test concepts are completely separate from train concepts
4. Reproducible with fixed random seed
"""

import xml.etree.ElementTree as ET
import random
import csv
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

RDF_FILE_PATH = "datasets/raw_datasets/ELSST_R5.rdf"
OUTPUT_DIR = "datasets/processed_datasets"
LANGUAGE = "en"
RANDOM_SEED = 42
TRAIN_RATIO = 0.70

# ============================================================================
# NAMESPACES
# ============================================================================

NAMESPACES = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
}


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_rdf_file(file_path):
    """Parse the RDF file."""
    print(f"Parsing: {file_path}")
    tree = ET.parse(file_path)
    return tree.getroot()


def extract_concepts(root):
    """Extract all SKOS concepts with English labels."""
    concepts = {}
    
    for desc in root.findall('.//rdf:Description', NAMESPACES):
        uri = desc.get(f'{{{NAMESPACES["rdf"]}}}about')
        if not uri:
            continue
        
        # Check if SKOS Concept
        type_elem = desc.find('rdf:type', NAMESPACES)
        if type_elem is None:
            continue
        type_res = type_elem.get(f'{{{NAMESPACES["rdf"]}}}resource')
        if type_res != 'http://www.w3.org/2004/02/skos/core#Concept':
            continue
        
        concept = {
            'prefLabel': None,
            'altLabels': [],
            'broader': [],
            'narrower': []
        }
        
        # Get prefLabel
        for pref in desc.findall('skos:prefLabel', NAMESPACES):
            lang = pref.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == LANGUAGE and pref.text:
                concept['prefLabel'] = pref.text
                break
        
        # Get altLabels
        for alt in desc.findall('skos:altLabel', NAMESPACES):
            lang = alt.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == LANGUAGE and alt.text:
                concept['altLabels'].append(alt.text)
        
        # Get broader (parents)
        for broader in desc.findall('skos:broader', NAMESPACES):
            parent_uri = broader.get(f'{{{NAMESPACES["rdf"]}}}resource')
            if parent_uri:
                concept['broader'].append(parent_uri)
        
        # Get narrower (children)
        for narrower in desc.findall('skos:narrower', NAMESPACES):
            child_uri = narrower.get(f'{{{NAMESPACES["rdf"]}}}resource')
            if child_uri:
                concept['narrower'].append(child_uri)
        
        if concept['prefLabel']:
            concepts[uri] = concept
    
    return concepts


# ============================================================================
# HIERARCHY FUNCTIONS
# ============================================================================

def get_all_ancestors(uri, concepts, visited=None):
    """Get all ancestors of a concept."""
    if visited is None:
        visited = set()
    
    ancestors = set()
    if uri in visited or uri not in concepts:
        return ancestors
    
    visited.add(uri)
    
    for parent in concepts[uri]['broader']:
        ancestors.add(parent)
        ancestors.update(get_all_ancestors(parent, concepts, visited.copy()))
    
    return ancestors


def get_all_descendants(uri, concepts, visited=None):
    """Get all descendants of a concept."""
    if visited is None:
        visited = set()
    
    descendants = set()
    if uri in visited or uri not in concepts:
        return descendants
    
    visited.add(uri)
    
    for child in concepts[uri]['narrower']:
        descendants.add(child)
        descendants.update(get_all_descendants(child, concepts, visited.copy()))
    
    return descendants


def build_exclusion_sets(concepts):
    """Build exclusion sets: self + ancestors + descendants."""
    exclusions = {}
    for uri in concepts:
        excluded = {uri}  # Self
        excluded.update(get_all_ancestors(uri, concepts))
        excluded.update(get_all_descendants(uri, concepts))
        exclusions[uri] = excluded
    return exclusions


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def get_all_terms(uri, concepts):
    """Get all terms (prefLabel + altLabels) for a concept."""
    terms = {concepts[uri]['prefLabel']}
    terms.update(concepts[uri]['altLabels'])
    return terms


def split_concepts_no_term_overlap(concepts, train_ratio, seed):
    """
    Split concepts into train and test sets ensuring NO term overlap.
    If a term appears in multiple concepts, all those concepts go to the same set.
    """
    random.seed(seed)
    
    # Build term -> concepts mapping
    term_to_concepts = {}
    for uri in concepts:
        for term in get_all_terms(uri, concepts):
            if term not in term_to_concepts:
                term_to_concepts[term] = set()
            term_to_concepts[term].add(uri)
    
    # Find concepts that share terms (must stay together)
    # Use union-find to group concepts
    parent = {uri: uri for uri in concepts}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union concepts that share any term
    for term, uris in term_to_concepts.items():
        uri_list = list(uris)
        for i in range(1, len(uri_list)):
            union(uri_list[0], uri_list[i])
    
    # Group concepts by their root
    groups = {}
    for uri in concepts:
        root = find(uri)
        if root not in groups:
            groups[root] = []
        groups[root].append(uri)
    
    # Shuffle groups and split
    group_list = list(groups.values())
    random.shuffle(group_list)
    
    train_uris = set()
    test_uris = set()
    total = len(concepts)
    target_train = int(total * train_ratio)
    
    for group in group_list:
        if len(train_uris) < target_train:
            train_uris.update(group)
        else:
            test_uris.update(group)
    
    return train_uris, test_uris


# ============================================================================
# PAIR GENERATION
# ============================================================================

def generate_positive_pairs(concept_uris, concepts):
    """
    Generate ALL positive pairs for concepts:
    1. prefLabel ↔ each altLabel (concept paired with its entry terms)
    2. altLabel ↔ altLabel (entry terms paired with each other)
    
    Example: Concept "WORK AT HOME" with altLabels ["OUTWORK", "HOME-BASED WORK"]
    Generates:
      - WORK AT HOME ↔ OUTWORK
      - WORK AT HOME ↔ HOME-BASED WORK
      - OUTWORK ↔ HOME-BASED WORK
    """
    pairs = []
    
    for uri in concept_uris:
        pref_label = concepts[uri]['prefLabel']
        alt_labels = concepts[uri]['altLabels']
        
        # 1. Pair prefLabel with each altLabel
        for alt in alt_labels:
            pairs.append({
                'term1': pref_label,
                'term2': alt,
                'concept': pref_label,
                'concept_uri': uri,
                'label': 1
            })
        
        # 2. Pair altLabels with each other
        for i in range(len(alt_labels)):
            for j in range(i + 1, len(alt_labels)):
                pairs.append({
                    'term1': alt_labels[i],
                    'term2': alt_labels[j],
                    'concept': pref_label,
                    'concept_uri': uri,
                    'label': 1
                })
    
    return pairs


def generate_negative_pairs(concept_uris, concepts, exclusion_sets):
    """
    Generate all valid negative pairs between concepts.
    Rules:
    - No self-pairing
    - No parent/ancestor relationships
    """
    pairs = []
    uri_list = sorted(concept_uris)  # Sort for reproducibility
    
    for i in range(len(uri_list)):
        uri1 = uri_list[i]
        excluded = exclusion_sets[uri1]
        label1 = concepts[uri1]['prefLabel']
        
        for j in range(i + 1, len(uri_list)):
            uri2 = uri_list[j]
            
            # Skip if hierarchically related
            if uri2 in excluded:
                continue
            
            label2 = concepts[uri2]['prefLabel']
            
            pairs.append({
                'term1': label1,
                'term2': label2,
                'concept1_uri': uri1,
                'concept2_uri': uri2,
                'label': 0
            })
    
    return pairs


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_to_csv(pairs, filename):
    """Save pairs to CSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not pairs:
        print(f"  No pairs to save for {filename}")
        return
    
    fieldnames = list(pairs[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    
    print(f"  Saved: {filepath} ({len(pairs)} rows)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ELSST TRAIN/TEST DATASET GENERATION")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Train Ratio: {TRAIN_RATIO}")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Parse and extract
    root = parse_rdf_file(RDF_FILE_PATH)
    concepts = extract_concepts(root)
    print(f"Extracted {len(concepts)} concepts")
    
    # Step 2: Build exclusion sets
    print("Building hierarchy exclusion sets...")
    exclusion_sets = build_exclusion_sets(concepts)
    
    # Step 3: Split concepts (ensuring NO term overlap)
    print("Splitting concepts into train/test (no term overlap)...")
    train_uris, test_uris = split_concepts_no_term_overlap(concepts, TRAIN_RATIO, RANDOM_SEED)
    print(f"  Train concepts: {len(train_uris)}")
    print(f"  Test concepts:  {len(test_uris)}")
    
    # Verify no concept overlap
    concept_overlap = train_uris & test_uris
    print(f"  Concept overlap: {len(concept_overlap)} (should be 0)")
    
    # Verify no term overlap
    train_terms = set()
    for uri in train_uris:
        train_terms.update(get_all_terms(uri, concepts))
    
    test_terms = set()
    for uri in test_uris:
        test_terms.update(get_all_terms(uri, concepts))
    
    term_overlap = train_terms & test_terms
    print(f"  Term overlap:    {len(term_overlap)} (should be 0)")
    
    if term_overlap:
        print(f"  WARNING: Overlapping terms: {list(term_overlap)[:5]}")
    
    # Step 4: Generate pairs for TRAIN
    print("\nGenerating TRAIN pairs...")
    train_positive = generate_positive_pairs(train_uris, concepts)
    train_negative = generate_negative_pairs(train_uris, concepts, exclusion_sets)
    
    # Step 5: Generate pairs for TEST
    print("Generating TEST pairs...")
    test_positive = generate_positive_pairs(test_uris, concepts)
    test_negative = generate_negative_pairs(test_uris, concepts, exclusion_sets)
    
    # Step 6: Save datasets
    print("\nSaving datasets...")
    save_to_csv(train_positive, "train_positive_pairs.csv")
    save_to_csv(train_negative, "train_negative_pairs.csv")
    save_to_csv(test_positive, "test_positive_pairs.csv")
    save_to_csv(test_negative, "test_negative_pairs.csv")
    
    # Step 7: Print summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n--- CONFIGURATION ---")
    print(f"Random Seed:           {RANDOM_SEED}")
    print(f"Train/Test Ratio:      {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}")
    
    print(f"\n--- CONCEPT & TERM SPLIT ---")
    print(f"Total concepts:        {len(concepts)}")
    print(f"Train concepts:        {len(train_uris)}")
    print(f"Test concepts:         {len(test_uris)}")
    print(f"Concept overlap:       {len(concept_overlap)} (verified: 0)")
    print(f"Train terms:           {len(train_terms)}")
    print(f"Test terms:            {len(test_terms)}")
    print(f"Term overlap:          {len(term_overlap)} (verified: 0)")
    
    # Count concepts with altLabels
    train_with_alts = sum(1 for u in train_uris if concepts[u]['altLabels'])
    test_with_alts = sum(1 for u in test_uris if concepts[u]['altLabels'])
    train_with_2plus = sum(1 for u in train_uris if len(concepts[u]['altLabels']) >= 2)
    test_with_2plus = sum(1 for u in test_uris if len(concepts[u]['altLabels']) >= 2)
    
    print(f"\n--- POSITIVE PAIRS ---")
    print(f"Train positive pairs:  {len(train_positive)}")
    print(f"  - prefLabel ↔ altLabel: from {train_with_alts} concepts with altLabels")
    print(f"  - altLabel ↔ altLabel:  from {train_with_2plus} concepts with 2+ altLabels")
    print(f"Test positive pairs:   {len(test_positive)}")
    print(f"  - prefLabel ↔ altLabel: from {test_with_alts} concepts with altLabels")
    print(f"  - altLabel ↔ altLabel:  from {test_with_2plus} concepts with 2+ altLabels")
    
    # Calculate excluded hierarchical pairs
    max_train = len(train_uris) * (len(train_uris) - 1) // 2
    max_test = len(test_uris) * (len(test_uris) - 1) // 2
    
    print(f"\n--- NEGATIVE PAIRS (unrelated concepts) ---")
    print(f"Train negative pairs:  {len(train_negative)}")
    print(f"  Max possible:        {max_train}")
    print(f"  Excluded (hierarchy): {max_train - len(train_negative)}")
    print(f"Test negative pairs:   {len(test_negative)}")
    print(f"  Max possible:        {max_test}")
    print(f"  Excluded (hierarchy): {max_test - len(test_negative)}")
    
    print(f"\n--- TOTAL ---")
    print(f"Train total pairs:     {len(train_positive) + len(train_negative)}")
    print(f"Test total pairs:      {len(test_positive) + len(test_negative)}")
    print(f"Grand total:           {len(train_positive) + len(train_negative) + len(test_positive) + len(test_negative)}")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print(f"  - train_positive_pairs.csv  ({len(train_positive)} pairs)")
    print(f"  - train_negative_pairs.csv  ({len(train_negative)} pairs)")
    print(f"  - test_positive_pairs.csv   ({len(test_positive)} pairs)")
    print(f"  - test_negative_pairs.csv   ({len(test_negative)} pairs)")
    print("=" * 70)


if __name__ == "__main__":
    main()