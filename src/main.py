import argparse
import sys
from parser import parse_json

def build_parser():
    parser = argparse.ArgumentParser(description='MLHub - PornHub but for ML')
    parser.add_argument('--model', required=True)
    return parser

def parse_args(parser):
    return parser.parse_args()

def main():
    parser = build_parser()
    args = parse_args(parser)
    parse_json(args.model)
    sys.exit(0)

if __name__ == '__main__':
    main()
