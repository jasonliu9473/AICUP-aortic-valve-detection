def check_duplicates(file_path, save_duplicates=True):
    seen = {}  # Track first occurrence line number
    duplicate_lines = []
    total_lines = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            
            total_lines += 1
            first_col = parts[0]
            
            if first_col in seen:
                duplicate_lines.append(line.strip())
#                 print(f"Duplicate found at line {line_num}: {first_col} (first seen at line {seen[first_col]})")
            else:
                seen[first_col] = line_num
    
    # Print summary
    unique_count = len(seen)
    duplicate_count = len(duplicate_lines)
    
    print("\nSummary:")
    print(f"Total entries processed: {total_lines}")
    print(f"Unique entries: {unique_count}")
    print(f"Duplicate occurrences: {duplicate_count}")
    
    # Optionally save duplicates
    if save_duplicates and duplicate_lines:
        with open("duplicates.txt", "w") as out:
            out.write("\n".join(duplicate_lines))
        print(f"\nDuplicate lines saved to duplicates.txt")

check_duplicates("predictions_file_name.txt")