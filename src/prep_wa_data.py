import os
import re
import argparse


def clean_up_text(input_file, output_file, specific_names=None, remove_specials=False):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    cleaned_lines = []

    for line in lines:
        # Remove media exclusions or any text enclosed in <>
        line = re.sub(r"<[^>]*>", "", line)

        # Remove leading timestamp and extract the name
        line = re.sub(r"^\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2} - ", "", line)

        # Remove Unicode characters
        line = line.encode("ascii", "ignore").decode("ascii")

        # Remove special characters based on the flag
        if remove_specials:
            # Remove all special symbols except :,()!?
            line = re.sub(r"[^\w\s:()!?]", "", line)
        else:
            # Remove emojis and non-standard special characters, retain common ones
            line = re.sub(r"[^\w\s:,.!?*()#-]", "", line)

        # Strip leading/trailing whitespace
        line = line.strip()

        # Ensure line is not empty and starts with a name
        if ":" in line:
            name_part, _, message_part = line.partition(":")
            message_part = message_part.strip()
            if message_part:
                if specific_names:
                    # Only keep lines for specific names
                    if name_part in specific_names:
                        cleaned_lines.append(f"{name_part}: {message_part}")
                else:
                    # Append any valid line
                    cleaned_lines.append(f"{name_part}: {message_part}")

    # Write cleaned content to the output file
    if output_file is None:
        print("\n".join(cleaned_lines))
    else:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(cleaned_lines))


def process_directory(input_dir, specific_names=None, out=None, remove_specials=False):
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    combined_lines = []

    for file in files:
        input_file = os.path.join(input_dir, file)
        if out:
            output_file = os.path.join(out, file.replace(".txt", "_prep.txt"))
        else:
            output_file = None

        if specific_names:
            # Collect lines for specific names
            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    # Remove media exclusions or any text enclosed in <>
                    line = re.sub(r"<[^>]*>", "", line)

                    # Remove leading timestamp and extract the name
                    line = re.sub(r"^\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2} - ", "", line)

                    # Remove Unicode characters
                    line = line.encode("ascii", "ignore").decode("ascii")

                    # Remove special characters based on the flag
                    if remove_specials:
                        # Remove all special symbols except :,()!?
                        line = re.sub(r"[^\w\s:()!?]", "", line)
                    else:
                        # Remove emojis and non-standard special characters, retain common ones
                        line = re.sub(r"[^\w\s:,.!?*()#-]", "", line)

                    # Strip leading/trailing whitespace
                    line = line.strip()

                    if ":" in line:
                        name_part, _, message_part = line.partition(":")
                        message_part = message_part.strip()
                        if message_part and name_part in specific_names:
                            combined_lines.append(f"{name_part}: {message_part}")
        else:
            clean_up_text(input_file, output_file, remove_specials=remove_specials)

    if specific_names:
        output_filename = f"{'_'.join(specific_names)}.txt"
        if out:
            combined_output_file = os.path.join(out, output_filename)
        else:
            combined_output_file = output_filename
        with open(combined_output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up chat logs in a directory.")
    parser.add_argument("input_dir", help="Directory with .txt files to process.")
    parser.add_argument(
        "--names", nargs="*", help="Specific names to filter (e.g., Karl)."
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Path to output directory."
    )
    parser.add_argument(
        "--remove-specials",
        action="store_true",
        help="Remove all special symbols except ':', '(', ')', '!', '?'.",
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        specific_names=args.names,
        out=args.output,
        remove_specials=args.remove_specials,
    )
