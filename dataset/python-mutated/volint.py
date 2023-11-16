"""
Script support for validating a VO file.
"""

def main(args=None):
    if False:
        return 10
    import argparse
    from . import table
    parser = argparse.ArgumentParser(description='Check a VOTable file for compliance to the VOTable specification')
    parser.add_argument('filename', nargs=1, help='Path to VOTable file to check')
    args = parser.parse_args(args)
    table.validate(args.filename[0])