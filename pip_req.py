input_file = "requirements_pip.txt"  # Plik wejściowy
output_file = "requirements_clean.txt"  # Poprawiony plik

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Pomijaj komentarze i puste linie
        if line.strip().startswith("#") or not line.strip():
            continue

        # Usuń fragmenty typu `=py39hd77b12b_8` oraz podwójne `==`
        clean_line = line.split("==")[0].strip()
        if "==" in line:
            clean_line += "==" + line.split("==")[1].split("=")[0].strip()

        # Pisz poprawioną linię do pliku wyjściowego
        outfile.write(clean_line + "\n")
