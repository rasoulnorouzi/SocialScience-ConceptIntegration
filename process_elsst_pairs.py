"""
ELSST RDF Data Processing Script
================================
This script processes the ELSST (European Language Social Science Thesaurus) RDF data
to create train/test splits of concepts and generate positive/negative pairs.

Strategy:
- Split CONCEPTS into train (70%) and test (30%)
- Generate positive pairs: entry term combinations within same concept
- Generate negative pairs: unrelated concepts (excluding hierarchical relations)
- All operations are reproducible with fixed random seed
"""

import xml.etree.ElementTree as ET
import random
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

RDF_FILE_PATH = "datasets/raw_datasets/ELSST_R5.rdf"
OUTPUT_DIR = "datasets/processed_datasets"
LANGUAGE = "en"  # Focus on English labels

# Reproducibility
RANDOM_SEED = 42

# Train/Test split ratio
TRAIN_RATIO = 0.70
TEST_RATIO = 0.30

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
    """Parse the RDF file and return the root element."""
    print(f"Parsing RDF file: {file_path}")
    tree = ET.parse(file_path)
    return tree.getroot()


def extract_concepts(root):
    """
    Extract all SKOS concepts from the RDF data.
    
    Returns:
        concepts: dict mapping concept URI to concept data
    """
    concepts = {}
    
    for description in root.findall('.//rdf:Description', NAMESPACES):
        uri = description.get(f'{{{NAMESPACES["rdf"]}}}about')
        
        if not uri:
            continue
            
        type_elem = description.find('rdf:type', NAMESPACES)
        if type_elem is None:
            continue
            
        type_resource = type_elem.get(f'{{{NAMESPACES["rdf"]}}}resource')
        if type_resource != 'http://www.w3.org/2004/02/skos/core#Concept':
            continue
        
        concept_data = {
            'prefLabel': None,
            'altLabels': [],
            'broader': [],
            'narrower': []
        }
        
        # Extract prefLabel for specified language
        for pref_label in description.findall('skos:prefLabel', NAMESPACES):
            lang = pref_label.get(f'{{{NAMESPACES["rdf"]}}}lang') or pref_label.get('{http://www.w3.org/XML/1998/namespace}lang')
            if lang == LANGUAGE:
                concept_data['prefLabel'] = pref_label.text
                break
        
        # Extract altLabels (entry terms)
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
        
        if concept_data['prefLabel']:
            concepts[uri] = concept_data
    
    return concepts


# ============================================================================
# HIERARCHY FUNCTIONS
# ============================================================================

def get_all_ancestors(concept_uri, concepts, visited=None):
    """Recursively get all ancestors of a concept."""
    if visited is None:
        visited = set()
    
    ancestors = set()
    
    if concept_uri in visited or concept_uri not in concepts:
        return ancestors
    
    visited.add(concept_uri)
    
    for parent_uri in concepts[concept_uri]['broader']:
        ancestors.add(parent_uri)
        ancestors.update(get_all_ancestors(parent_uri, concepts, visited.copy()))
    
    return ancestors


def get_all_descendants(concept_uri, concepts, visited=None):
    """Recursively get all descendants of a concept."""
    if visited is None:
        visited = set()
    
    descendants = set()
    
    if concept_uri in visited or concept_uri not in concepts:
        return descendants
    
    visited.add(concept_uri)
    
    for child_uri in concepts[concept_uri]['narrower']:
        descendants.add(child_uri)
        descendants.update(get_all_descendants(child_uri, concepts, visited.copy()))
    
    return descendants


def build_exclusion_sets(concepts, concept_uris):
    """
    Build exclusion sets for a subset of concepts.
    A concept cannot be paired with itself, ancestors, or descendants.
    """
    exclusion_sets = {}
    concept_set = set(concept_uris)
    
    for uri in concept_uris:
        excluded = set()
        excluded.add(uri)  # Exclude self
        
        # Get ancestors and descendants (from full concept set for accurate hierarchy)
        ancestors = get_all_ancestors(uri, concepts)
        descendants = get_all_descendants(uri, concepts)
        
        # Only include exclusions that are in our subset
        excluded.update(ancestors & concept_set)
        excluded.update(descendants & concept_set)
        
        exclusion_sets[uri] = excluded
    
    return exclusion_sets


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def split_concepts(concepts, train_ratio, seed):
    """
    Split concepts into train and test sets.
    
    Args:
        concepts: dict of all concepts
        train_ratio: ratio for training set (e.g., 0.70)
        seed: random seed for reproducibility
    
    Returns:
        train_uris: list of URIs for training
        test_uris: list of URIs for testing
    """
    random.seed(seed)
    
    all_uris = list(concepts.keys())
    random.shuffle(all_uris)
    
    split_idx = int(len(all_uris) * train_ratio)
    train_uris = all_uris[:split_idx]
    test_uris = all_uris[split_idx:]
    
    return train_uris, test_uris


# ============================================================================
# PAIR GENERATION FUNCTIONS
# ============================================================================

def generate_positive_pairs(concepts, concept_uris):
    """
    Generate positive pairs from entry terms (altLabels) of the same concept.
    Only considers concepts in the given subset.
    """
    positive_pairs = []
    
    for uri in concept_uris:
        data = concepts[uri]
        pref_label = data['prefLabel']
        alt_labels = data['altLabels']
        
        # Generate pairs between altLabels only
        for i in range(len(alt_labels)):
            for j in range(i + 1, len(alt_labels)):
                positive_pairs.append({
                    'term1': alt_labels[i],
                    'term2': alt_labels[j],
                    'concept_uri': uri,
                    'concept_prefLabel': pref_label,
                    'label': 1
                })
    
    return positive_pairs


def generate_negative_pairs(concepts, concept_uris, exclusion_sets, set_name=""):
    """
    Generate all negative pairs between unrelated concepts in the subset.
    Excludes hierarchical relationships.
    """
    negative_pairs = []
    sorted_uris = sorted(concept_uris)
    total = len(sorted_uris)
    
    print(f"  Generating negative pairs for {set_name}...")
    
    for i in range(total):
        uri1 = sorted_uris[i]
        excluded = exclusion_sets[uri1]
        label1 = concepts[uri1]['prefLabel']
        
        for j in range(i + 1, total):
            uri2 = sorted_uris[j]
            
            if uri2 in excluded:
                continue
            
            label2 = concepts[uri2]['prefLabel']
            
            negative_pairs.append({
                'term1': label1,
                'term2': label2,
                'concept1_uri': uri1,
                'concept2_uri': uri2,
                'label': 0
            })
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{total} concepts...")
    
    return negative_pairs


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_to_json(data, filename):
    """Save data to JSON file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {filepath}")
    return filepath


def print_split_statistics(name, concepts, concept_uris, positive_pairs, negative_pairs):
    """Print statistics for a split."""
    concepts_with_alt = sum(1 for uri in concept_uris if len(concepts[uri]['altLabels']) >= 2)
    total_alt = sum(len(concepts[uri]['altLabels']) for uri in concept_uris)
    
    print(f"\n  --- {name.upper()} SET ---")
    print(f"  Concepts:                {len(concept_uris)}")
    print(f"  Concepts with 2+ alts:   {concepts_with_alt}")
    print(f"  Total altLabels:         {total_alt}")
    print(f"  Positive pairs:          {len(positive_pairs)}")
    print(f"  Negative pairs:          {len(negative_pairs)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("ELSST RDF DATA PROCESSING - TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Random Seed:    {RANDOM_SEED}")
    print(f"  Train Ratio:    {TRAIN_RATIO}")
    print(f"  Test Ratio:     {TEST_RATIO}")
    print(f"  Language:       {LANGUAGE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Parse RDF and extract concepts
    print("\n" + "-" * 70)
    print("Step 1: Extracting concepts from RDF")
    print("-" * 70)
    root = parse_rdf_file(RDF_FILE_PATH)
    concepts = extract_concepts(root)
    print(f"Extracted {len(concepts)} concepts with English labels")
    
    # Step 2: Split concepts into train/test
    print("\n" + "-" * 70)
    print("Step 2: Splitting concepts into train/test")
    print("-" * 70)
    train_uris, test_uris = split_concepts(concepts, TRAIN_RATIO, RANDOM_SEED)
    print(f"Train concepts: {len(train_uris)}")
    print(f"Test concepts:  {len(test_uris)}")
    
    # Step 3: Build exclusion sets for each split
    print("\n" + "-" * 70)
    print("Step 3: Building exclusion sets for hierarchy")
    print("-" * 70)
    train_exclusions = build_exclusion_sets(concepts, train_uris)
    test_exclusions = build_exclusion_sets(concepts, test_uris)
    print("Exclusion sets built for train and test")
    
    # Step 4: Generate pairs for training set
    print("\n" + "-" * 70)
    print("Step 4: Generating training pairs")
    print("-" * 70)
    train_positive = generate_positive_pairs(concepts, train_uris)
    print(f"Generated {len(train_positive)} positive pairs for training")
    
    train_negative = generate_negative_pairs(concepts, train_uris, train_exclusions)
    print(f"Generated {len(train_negative)} negative pairs for training")
    
    # Step 5: Generate pairs for test set
    print("\n" + "-" * 70)
    print("Step 5: Generating test pairs")
    print("-" * 70)
    test_positive = generate_positive_pairs(concepts, test_uris)
    print(f"Generated {len(test_positive)} positive pairs for testing")
    
    test_negative = generate_negative_pairs(concepts, test_uris, test_exclusions)
    print(f"Generated {len(test_negative)} negative pairs for testing")
    
    # Step 6: Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print_split_statistics("train", concepts, train_uris, train_positive, train_negative)
    print_split_statistics("test", concepts, test_uris, test_positive, test_negative)
    
    # Step 7: Save datasets
    print("\n" + "-" * 70)
    print("Step 6: Saving datasets")
    print("-" * 70)
    
    # Save train data
    save_to_json(train_positive, "train_positive_pairs.json")
    save_to_json(train_negative, "train_negative_pairs.json")
    
    # Save test data
    save_to_json(test_positive, "test_positive_pairs.json")
    save_to_json(test_negative, "test_negative_pairs.json")
    
    # Save concept lists for reference
    train_concepts_data = {
        uri: {
            'prefLabel': concepts[uri]['prefLabel'],
            'altLabels': concepts[uri]['altLabels']
        }
        for uri in train_uris
    }
    test_concepts_data = {
        uri: {
            'prefLabel': concepts[uri]['prefLabel'],
            'altLabels': concepts[uri]['altLabels']
        }
        for uri in test_uris
    }
    
    save_to_json(train_concepts_data, "train_concepts.json")
    save_to_json(test_concepts_data, "test_concepts.json")
    
    # Save split metadata
    metadata = {
        'random_seed': RANDOM_SEED,
        'train_ratio': TRAIN_RATIO,
        'test_ratio': TEST_RATIO,
        'language': LANGUAGE,
        'total_concepts': len(concepts),
        'train_concepts': len(train_uris),
        'test_concepts': len(test_uris),
        'train_positive_pairs': len(train_positive),
        'train_negative_pairs': len(train_negative),
        'test_positive_pairs': len(test_positive),
        'test_negative_pairs': len(test_negative)
    }
    save_to_json(metadata, "dataset_metadata.json")
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    return concepts, train_uris, test_uris


if __name__ == "__main__":
    main()
