#!/usr/bin/env python3
"""Verify every published raw archive and member against its SHA256 manifest."""
from raw_data import entries, verify


def main():
    records = entries()
    for row in records:
        verify(row, members=True)
        print(row['path']+': verified', flush=True)
    print(f'Verified {len(records)} original-record archives.')


if __name__ == '__main__':
    main()
