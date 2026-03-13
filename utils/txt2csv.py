#!/usr/bin/env python3
"""
Simple converter: edge-list TXT (with comment lines starting with '#') to CSV.

Usage:
	python utils/txt2csv.py input.txt [output.csv]

The script skips lines starting with '#' or empty lines and writes the first two
columns of each data line to the CSV file.
"""
import argparse
import csv
import os
import sys


def convert(input_path, output_path=None, delimiter=',', header=False):
	if output_path is None:
		base, _ = os.path.splitext(input_path)
		output_path = base + '.csv'

	with open(input_path, 'r', encoding='utf-8') as fin, \
		 open(output_path, 'w', newline='', encoding='utf-8') as fout:
		writer = csv.writer(fout, delimiter=delimiter)
		if header:
			writer.writerow(['u', 'v'])

		for lineno, line in enumerate(fin, start=1):
			s = line.strip()
			if not s or s.startswith('#'):
				continue
			parts = s.split()
			if len(parts) < 2:
				sys.stderr.write(f"Skipping malformed line {lineno}: {s}\n")
				continue
			writer.writerow(parts[:2])

	print(f"Wrote {output_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Convert edge-list TXT (with '#' comments) to CSV"
	)
	parser.add_argument('input', help='Input txt file')
	parser.add_argument('output', nargs='?', help='Output csv file (default: same name .csv)')
	parser.add_argument('--delimiter', '-d', default=',', help='CSV delimiter (default: ,)')
	parser.add_argument('--header', action='store_true', help='Write header row "u,v"')
	parser.add_argument('--matrix', action='store_true', help='Output adjacency matrix CSV (rows=|U|, cols=|V|)')
	parser.add_argument('--rows', type=int, help='Number of rows for matrix (override or when no header)')
	parser.add_argument('--cols', type=int, help='Number of columns for matrix (override or when no header)')
	args = parser.parse_args()

	if not os.path.exists(args.input):
		print(f"Input file not found: {args.input}", file=sys.stderr)
		sys.exit(2)

	if args.matrix:
		# Build adjacency matrix
		convert_to_matrix(args.input, args.output, args.delimiter, args.header, args.rows, args.cols)
	else:
		convert(args.input, args.output, args.delimiter, args.header)


def convert_to_matrix(input_path, output_path=None, delimiter=',', write_header=False, rows_arg=None, cols_arg=None):
	# First pass: detect |U| and |V| from comments, and collect edges
	u_size = None
	v_size = None
	edges = []
	with open(input_path, 'r', encoding='utf-8') as fin:
		for lineno, line in enumerate(fin, start=1):
			s = line.strip()
			if not s:
				continue
			if s.startswith('#'):
				# look for patterns like: # |U|: 100
				if '|U|' in s and ':' in s:
					try:
						u_size = int(s.split(':', 1)[1].strip())
					except Exception:
						pass
				if '|V|' in s and ':' in s:
					try:
						v_size = int(s.split(':', 1)[1].strip())
					except Exception:
						pass
				continue
			parts = s.split()
			if len(parts) < 2:
				sys.stderr.write(f"Skipping malformed line {lineno}: {s}\n")
				continue
			try:
				u = int(parts[0])
				v = int(parts[1])
			except ValueError:
				sys.stderr.write(f"Non-integer indices on line {lineno}: {s}\n")
				continue
			edges.append((u, v))

	# Override with explicit args if provided
	if rows_arg is not None:
		u_size = rows_arg
	if cols_arg is not None:
		v_size = cols_arg

	# If sizes still unknown, infer from max indices
	if u_size is None or v_size is None:
		max_u = max((u for u, _ in edges), default=-1)
		max_v = max((v for _, v in edges), default=-1)
		if u_size is None:
			u_size = max_u + 1
		if v_size is None:
			v_size = max_v + 1

	if u_size is None or v_size is None:
		print("Could not determine matrix dimensions (no header and no edges)", file=sys.stderr)
		sys.exit(2)

	# Initialize matrix of zeros
	matrix = [[0] * v_size for _ in range(u_size)]
	for u, v in edges:
		if 0 <= u < u_size and 0 <= v < v_size:
			matrix[u][v] = 1
		else:
			sys.stderr.write(f"Edge ({u},{v}) out of bounds for matrix {u_size}x{v_size}\n")

	if output_path is None:
		base, _ = os.path.splitext(input_path)
		output_path = base + '.csv'

	with open(output_path, 'w', newline='', encoding='utf-8') as fout:
		writer = csv.writer(fout, delimiter=delimiter)
		for row in matrix:
			writer.writerow(row)

	print(f"Wrote adjacency matrix to {output_path} ({u_size}x{v_size})")


if __name__ == '__main__':
	main()

