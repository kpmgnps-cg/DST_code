# Defining dictionary for extracting gegraphical features from the text
import re

regexDictionary = {}
regexDictionary['street_address'] = re.compile(
    '\d{1,4} [\w\s]{1,24}(?:STREET|ST|AVENUE|SUITE|VILLAGE|AVE|ROAD|RD|HIGHWAY|HWY|SQUARE|VALLEY|SQ|TRAIL|TRL|DRIVE|DR|COURT|CT|PARK|PARKWAY|PKWY|CIRCLE|CIR|BOULEVARD|BLVD|PKWY STE|STE|ST|APT|RD|ROAD|PARK|P O BOX|VALLEY FORGE})\W?(?=\s|$)',
    re.IGNORECASE)
regexDictionary['zip_code'] = re.compile(r'\b\d{4,5}(?:[-\s]\d{4})?\b')
regexDictionary['zip_code2'] = re.compile(r'\b\d{4,5}(?:[-\s]\d{4})\b')
regexDictionary['po_box'] = re.compile(r'(P\.? ?O\.? BOX \d+|PO BOX \d+|P O BOX|P.O. BOX|POBOX \d+)', re.IGNORECASE)
regexDictionary['ste_addr'] = re.compile(r'STE \d+', re.IGNORECASE)

