import json
import config
import re

def extract_boundary_names(filename):
    """
    Extract boundary condition names from a fluent.msh mesh file
    Args:
        filename (str): Path to the text file to process
    Returns:
        list: Filtered result list
    """
    # Read file content
    with open(filename, 'r') as f:
        content = f.read().splitlines()

    # Find the starting point
    start_index = -1
    for i in range(len(content)-1, -1, -1):  # Search for the starting line from back to front
        if content[i].strip() == '(0 "Zone Sections")':
            start_index = i
            break
    
    if start_index == -1:
        return []

    # Extract relevant lines
    pattern = re.compile(r'\(\d+\s+\(\d+\s+\S+\s+(\S+)\)\(\)\)')
    results = []
    
    # Process each line of data
    for line in content[start_index+1:]:
        line = line.strip()
        if not line.startswith('(39'):
            continue
            
        match = pattern.match(line)
        if match:
            value = match.group(1)
            # Filter *_FLUID and *_SOLID
            if not re.search(r'^(FLUID|\w+?_FLUID|\w+?_SOLID)$', value):
                results.append(value)
    
    config.case_boundaries = results
