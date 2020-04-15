from rdkit.Chem import AllChem as Chem


def main():
    import argparse
    parser = argparse.ArgumentParser(description='tSNE analysis')
    parser.add_argument('-i', '--inchi', type=str, help='Input inchi string')
    args = parser.parse_args()

    inchi = args.inchi
    mol = Chem.MolFromInchi(inchi)
    inchi = Chem.MolToInchi(mol)


if __name__ == '__main__':
    main()
