try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics, build_molecule
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False
import qm9.dataset as dataset
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats
from qm9 import bond_analyze


# 'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],

analyzed_19 ={'atom_types': {1: 93818, 3: 21212, 0: 139496, 2: 8251, 4: 26},
            'distances': [0, 0, 0, 0, 0, 0, 0, 22566, 258690, 16534, 50256, 181302, 19676, 122590, 23874, 54834, 309290, 205426, 172004, 229940, 193180, 193058, 161294, 178292, 152184, 157242, 189186, 150298, 125750, 147020, 127574, 133654, 142696, 125906, 98168, 95340, 88632, 80694, 71750, 64466, 55740, 44570, 42850, 36084, 29310, 27268, 23696, 20254, 17112, 14130, 12220, 10660, 9112, 7640, 6378, 5350, 4384, 3650, 2840, 2362, 2050, 1662, 1414, 1216, 966, 856, 492, 516, 420, 326, 388, 326, 236, 140, 130, 92, 62, 52, 78, 56, 24, 8, 10, 12, 18, 2, 10, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}


class Histogram_discrete:
    def __init__(self, name='histogram'):
        self.name = name
        self.bins = {}

    def add(self, elements):
        for e in elements:
            if e in self.bins:
                self.bins[e] += 1
            else:
                self.bins[e] = 1

    def normalize(self):
        total = 0.
        for key in self.bins:
            total += self.bins[key]
        for key in self.bins:
            self.bins[key] = self.bins[key] / total

    def plot(self, save_path=None):
        width = 1  # the width of the bars
        fig, ax = plt.subplots()
        x, y = [], []
        for key in self.bins:
            x.append(key)
            y.append(self.bins[key])

        ax.bar(x, y, width)
        plt.title(self.name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


class Histogram_cont:
    def __init__(self, num_bins=100, range=(0., 13.), name='histogram', ignore_zeros=False):
        self.name = name
        self.bins = [0] * num_bins
        self.range = range
        self.ignore_zeros = ignore_zeros

    def add(self, elements):
        for e in elements:
            if not self.ignore_zeros or e > 1e-8:
                i = int(float(e) / self.range[1] * len(self.bins))
                i = min(i, len(self.bins) - 1)
                self.bins[i] += 1

    def plot(self, save_path=None):
        width = (self.range[1] - self.range[0])/len(self.bins)                 # the width of the bars
        fig, ax = plt.subplots()

        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1] + width / 2
        ax.bar(x, self.bins, width)
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    def plot_both(self, hist_b, save_path=None, wandb=None):
        ## TO DO: Check if the relation of bins and linspace is correct
        hist_a = normalize_histogram(self.bins)
        hist_b = normalize_histogram(hist_b)

        #width = (self.range[1] - self.range[0]) / len(self.bins)  # the width of the bars
        fig, ax = plt.subplots()
        x = np.linspace(self.range[0], self.range[1], num=len(self.bins) + 1)[:-1]
        ax.step(x, hist_b)
        ax.step(x, hist_a)
        ax.legend(['True', 'Learned'])
        plt.title(self.name)

        if save_path is not None:
            plt.savefig(save_path)
            if wandb is not None:
                if wandb is not None:
                    # Log image(s)
                    im = plt.imread(save_path)
                    wandb.log({save_path: [wandb.Image(im, caption=save_path)]})
        else:
            plt.show()
        plt.close()


def normalize_histogram(hist):
    hist = np.array(hist)
    prob = hist / np.sum(hist)
    return prob


def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    dist = dist.flatten()
    return dist


def earth_mover_distance(h1, h2):
    p1 = normalize_histogram(h1)
    p2 = normalize_histogram(h2)
    distance = sp_stats.wasserstein_distance(p1, p2)
    return distance


def kl_divergence(p1, p2):
    return np.sum(p1*np.log(p1 / p2))


def kl_divergence_sym(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10
    kl = kl_divergence(p1, p2)
    kl_flipped = kl_divergence(p2, p1)
    return (kl + kl_flipped) / 2.


def js_divergence(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10
    M = (p1 + p2)/2
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2
    return js


def main_analyze_qm9(remove_h: bool, dataset_name='qm9', n_atoms=None):
    class DataLoaderConfig(object):
        def __init__(self):
            self.batch_size = 128
            self.remove_h = remove_h
            self.filter_n_atoms = n_atoms
            self.num_workers = 0
            self.include_charges = True
            self.dataset = dataset_name  #could be qm9, qm9_first_half or qm9_second_half
            self.datadir = 'qm9/temp'

    cfg = DataLoaderConfig()

    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

    hist_nodes = Histogram_discrete('Histogram # nodes')
    hist_atom_type = Histogram_discrete('Histogram of atom types')
    hist_dist = Histogram_cont(name='Histogram relative distances', ignore_zeros=True)

    for i, data in enumerate(dataloaders['train']):
        print(i * cfg.batch_size)

        # Histogram num_nodes
        num_nodes = torch.sum(data['atom_mask'], dim=1)
        num_nodes = list(num_nodes.numpy())
        hist_nodes.add(num_nodes)

        #Histogram edge distances
        x = data['positions'] * data['atom_mask'].unsqueeze(2)
        dist = coord2distances(x)
        hist_dist.add(list(dist.numpy()))

        # Histogram of atom types
        one_hot = data['one_hot'].double()
        atom = torch.argmax(one_hot, 2)
        atom = atom.flatten()
        mask = data['atom_mask'].flatten()
        masked_atoms = list(atom[mask].numpy())
        hist_atom_type.add(masked_atoms)

    hist_dist.plot()
    hist_dist.plot_both(hist_dist.bins[::-1])
    print("KL divergence A %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins[::-1]))
    print("KL divergence B %.4f" % kl_divergence_sym(hist_dist.bins, hist_dist.bins))
    print(hist_dist.bins)
    hist_nodes.plot()
    print("Histogram of the number of nodes", hist_nodes.bins)
    hist_atom_type.plot()
    print(" Histogram of the atom types (H (optional), C, N, O, F)", hist_atom_type.bins)


############################
# Validity and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


def process_loader(dataloader):
    """ Mask atoms, return positions and atom types"""
    out = []
    for data in dataloader:
        for i in range(data['positions'].size(0)):
            positions = data['positions'][i].view(-1, 3)
            one_hot = data['one_hot'][i].view(-1, 5).type(torch.float32)
            mask = data['atom_mask'][i].flatten()
            positions, one_hot = positions[mask], one_hot[mask]
            atom_type = torch.argmax(one_hot, dim=1)
            out.append((positions, atom_type))
    return out


def main_check_stability(remove_h: bool, batch_size=32):
    from configs import datasets_config
    import qm9.dataset as dataset

    class Config:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = 0
            self.remove_h = remove_h
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.dataset = 'qm9'
            self.include_charges = True
            self.filter_molecule_size = None
            self.sequential = False

    cfg = Config()

    dataset_info = datasets_config.qm9_with_h
    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)
    if use_rdkit:
        from qm9.rdkit_functions import BasicMolecularMetrics
        metrics = BasicMolecularMetrics(dataset_info)

    atom_decoder = dataset_info['atom_decoder']

    def test_validity_for(dataloader):
        count_mol_stable = 0
        count_atm_stable = 0
        count_mol_total = 0
        count_atm_total = 0
        for [positions, atom_types] in dataloader:
            is_stable, nr_stable, total = check_stability(
                positions, atom_types, dataset_info)

            count_atm_stable += nr_stable
            count_atm_total += total

            count_mol_stable += int(is_stable)
            count_mol_total += 1

            print(f"Stable molecules "
                  f"{100. * count_mol_stable/count_mol_total:.2f} \t"
                  f"Stable atoms: "
                  f"{100. * count_atm_stable/count_atm_total:.2f} \t"
                  f"Counted molecules {count_mol_total}/{len(dataloader)*batch_size}")

    train_loader = process_loader(dataloaders['train'])
    test_loader = process_loader(dataloaders['test'])
    if use_rdkit:
        print('For test')
        metrics.evaluate(test_loader)
        print('For train')
        metrics.evaluate(train_loader)
    else:
        print('For train')
        test_validity_for(train_loader)
        print('For test')
        test_validity_for(test_loader)


def analyze_stability_for_molecules(molecule_list, dataset_info):
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    if use_rdkit:
        metrics = BasicMolecularMetrics(dataset_info)
        rdkit_metrics = metrics.evaluate(processed_list)
        #print("Unique molecules:", rdkit_metrics[1])
        return validity_dict, rdkit_metrics
    else:
        return validity_dict, None


def analyze_node_distribution(mol_list, save_path):
    hist_nodes = Histogram_discrete('Histogram # nodes (stable molecules)')
    hist_atom_type = Histogram_discrete('Histogram of atom types')

    for molecule in mol_list:
        positions, atom_type = molecule
        hist_nodes.add([positions.shape[0]])
        hist_atom_type.add(atom_type)
    print("Histogram of #nodes")
    print(hist_nodes.bins)
    print("Histogram of # atom types")
    print(hist_atom_type.bins)
    hist_nodes.normalize()

def save_sdf_file(output_sdf_path, molecule_list, dataset_info):
    # data process from analyze_stability_for_molecules
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    # for mol in processed_list:
    #     pos, atom_type = mol
    #     validity_results = check_stability(pos, atom_type, dataset_info)

    #     molecule_stable += int(validity_results[0])
    #     nr_stable_bonds += int(validity_results[1])
    #     n_atoms += int(validity_results[2])

    # # Validity
    # fraction_mol_stable = molecule_stable / float(n_samples)
    # fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    # validity_dict = {
    #     'mol_stable': fraction_mol_stable,
    #     'atm_stable': fraction_atm_stable,
    # }
    sdf_writer = Chem.SDWriter(output_sdf_path)
    for graph in processed_list:
        mol = build_molecule(*graph, dataset_info)
        if mol is not None:  # Ensure the molecule is not None before writing
            # Optionally add dataset information as properties to the molecules
            for key, value in dataset_info.items():
                mol.SetProp(key, str(value))
            
            # Step 3: Write the molecule to the SDF file
            sdf_writer.write(mol)
    
    # Step 4: Close the SDF writer
    sdf_writer.close()
    
def compute_fcd_score(molecule_list, dataset_info):
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']
    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    processed_list = []
    molecules = []
    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))
    for graph in processed_list:
        mol = build_molecule(*graph, dataset_info)
        molecules.append(mol)


    refer = Chem.SDMolSupplier('./qm9/temp/qm9/qm9_test_output.sdf')
    # Filter out invalid molecules
    mols = [mol for mol in molecules if mol is not None]
    ref_mols = [mol for mol in refer if mol is not None]
    # Check the number of valid molecules
    valid_mols_count = len(mols)
    # print(f"Number of valid molecules: {valid_mols_count}")
    from fcd_torch import FCD
    fcd = FCD(device='cuda:0', n_jobs=8)
    smiles_list1 = []
    smiles_list2 = []
    for i, mol in enumerate(ref_mols):
        try:
            # mol = Chem.AddHs(mol)
            if mol:
                Chem.SanitizeMol(mol)  #
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            smiles_list2.append(smiles)
        except Exception as e:
            # print(f"Failed to process molecule {i}: {e}")
            continue
    pgen = fcd.precalc(smiles_list2)
    for i, mol in enumerate(mols):
        try:
            # mol = Chem.AddHs(mol)
            if mol:
                Chem.SanitizeMol(mol)  
            # success = AllChem.EmbedMolecule(mol)
            # if success == -1:
            #     AllChem.MMFFOptimizeMolecule(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            smiles_list1.append(smiles)
        except Exception as e:
            continue
    fcd_score = fcd(smiles_list1, pgen=pgen)
    print("FCD score: ", fcd_score," over %d valid generated mols"%(len(smiles_list1)))
    return fcd_score



import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def _is_valid_float(value):
    """Check if a value is a valid float (not None, not NaN, not inf)"""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return not (np.isnan(value) or np.isinf(value))
    return False

def calc_energy(mol, per_atom=False):
    """Calculate energy for a molecule using RDKit MMFF force field"""
    if mol is None:
        return None
    
    try:
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(mol)
        
        # Sanitize the molecule first to ensure valence is calculated
        Chem.SanitizeMol(mol_copy)
        # Add hydrogens
        mol_with_h = Chem.AddHs(mol_copy)
        # Generate 3D coordinates if not present
        if mol_with_h.GetNumConformers() == 0:
            print('mol_with_h has no conformers')
            success = AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            if success != 0:
                return None
        
        # Get MMFF molecule properties first
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
        if mmff_props is None:
            return None
        
        # Setup MMFF force field with properties
        ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, mmff_props)
        if ff is None:
            return None
            
        energy = ff.CalcEnergy()
        
        if per_atom and mol_with_h.GetNumAtoms() > 0:
            energy = energy / mol_with_h.GetNumAtoms()
            
        return energy if _is_valid_float(energy) else None
        
    except Exception as e:
        # print(f"Error in calc_energy: {str(e)}")
        return None

def optimise_mol(mol):
    """Optimize molecule geometry using RDKit MMFF"""
    if mol is None:
        return None
        
    try:
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(mol)
        
        # Sanitize the molecule first to ensure valence is calculated
        Chem.SanitizeMol(mol_copy)
        
        # Add hydrogens
        mol_with_h = Chem.AddHs(mol_copy)
        
        # Generate 3D coordinates if not present
        if mol_with_h.GetNumConformers() == 0:
            success = AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            if success != 0:
                return None
        
        # Get MMFF molecule properties first
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
        if mmff_props is None:
            return None
        
        success = AllChem.MMFFOptimizeMolecule(mol_with_h, mmffVariant='MMFF94s')
        
        if success != 0:
            # Optimization failed
            return None
            
        return mol_with_h
        
    except Exception as e:
        # print(f"Error in optimise_mol: {str(e)}")
        return None

def conf_distance(mol1, mol2):
    """Calculate RMSD between two molecules"""
    if mol1 is None or mol2 is None:
        return None
    try:
        # Both molecules should have conformers
        if mol1.GetNumConformers() == 0 or mol2.GetNumConformers() == 0:
            print('mol1 or mol2 has no conformers', mol1.GetNumConformers(), mol2.GetNumConformers())
            return None
        
        # To compare conformers, the atom count must match.
        # The hydrogens added in optimise_mol must be accounted for.
        # A simple way is to remove hydrogens from the optimized molecule before comparison.
        mol1_no_h = Chem.RemoveHs(mol1)
        mol2_no_h = Chem.RemoveHs(mol2)

        if mol1_no_h.GetNumAtoms() != mol2_no_h.GetNumAtoms():
            print("Atom count mismatch after removing H, cannot calculate RMSD.")
            return None
        
        # Calculate RMSD on the heavy-atom scaffolds
        rmsd = AllChem.GetBestRMS(mol1_no_h, mol2_no_h)
        return rmsd if _is_valid_float(rmsd) else None
    except Exception as e:
        print(str(e))
        return None

def compute_energy(molecule_list, dataset_info, optimise=False, per_atom=True):
    """
    Compute average energy for generated molecules
    
    Args:
        molecule_list: Dictionary containing 'one_hot', 'x', 'node_mask'
        dataset_info: Dataset information for molecule building
        optimise: Whether to optimize molecules before energy calculation
        per_atom: Whether to return per-atom energy
    
    Returns:
        Average energy as float, or None if no valid energies
    """
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']
    
    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    # Process molecules similar to compute_fcd_score
    processed_list = []
    molecules = []
    
    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))
    
    for graph in processed_list:
        mol = build_molecule(*graph, dataset_info)
        molecules.append(mol)

    # Filter valid molecules
    mols = [mol for mol in molecules if mol is not None]
    if not mols:
        print("No valid molecules for energy calculation")
        return None
    # Optimize if requested
    if optimise:
        mols = [optimise_mol(mol) for mol in mols]
        mols = [mol for mol in mols if mol is not None]
    
    # Calculate energies
    energies = [calc_energy(mol, per_atom=per_atom) for mol in mols]
    valid_energies = [energy for energy in energies if _is_valid_float(energy)]
    
    if not valid_energies:
        print("No valid energies calculated")
        return None
    
    average_energy = sum(valid_energies) / len(valid_energies)
    print(f"Average energy: {average_energy:.4f} over {len(valid_energies)} valid molecules")
    print(f"Median energy: {np.median(valid_energies):.4f}")
    
    return average_energy

def compute_rmsd(molecule_list, dataset_info):
    """
    Compute average RMSD between molecules and their optimized poses
    
    Args:
        molecule_list: Dictionary containing 'one_hot', 'x', 'node_mask'
        dataset_info: Dataset information for molecule building
    
    Returns:
        Average RMSD as float, or None if no valid RMSDs
    """
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']
    
    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    # Process molecules similar to compute_fcd_score
    processed_list = []
    molecules = []
    
    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))
    
    for graph in processed_list:
        mol = build_molecule(*graph, dataset_info)
        molecules.append(mol)

    # Filter valid molecules and get their indices
    valid_mols_with_idx = [(idx, mol) for idx, mol in enumerate(molecules) if mol is not None]
    
    if not valid_mols_with_idx:
        print("No valid molecules for RMSD calculation")
        return None

    # Optimize molecules
    opt_results = []
    for idx, mol in valid_mols_with_idx:
        opt_mol = optimise_mol(mol)
        if opt_mol is not None:
            opt_results.append((idx, mol, opt_mol))
    
    if not opt_results:
        print("No successfully optimized molecules")
        return None

    # Calculate RMSDs
    rmsds = []
    for idx, original_mol, opt_mol in opt_results:
        rmsd = conf_distance(original_mol, opt_mol)
        if _is_valid_float(rmsd):
            rmsds.append(rmsd)
    
    if not rmsds:
        print("No valid RMSDs calculated")
        return None
    
    average_rmsd = sum(rmsds) / len(rmsds)
    print(f"Average optimization RMSD: {average_rmsd:.4f} over {len(rmsds)} valid molecules")
    
    return average_rmsd

def compute_strain_energy(molecule_list, dataset_info, per_atom=False):
    """
    Computes the average strain energy for a list of generated molecules.

    The strain energy is the difference between the energy of the initial generated
    pose and the energy of the force-field-optimized pose.

    Args:
        molecule_list (dict): Dictionary containing 'one_hot', 'x', and 'node_mask'.
        dataset_info (dict): Dataset information for molecule building.
        per_atom (bool): Whether to calculate the energy on a per-atom basis.

    Returns:
        float: The average strain energy, or None if it could not be computed for any molecule.
    """
    # 1. Build initial molecules from the input data
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]
    n_samples = len(x)

    initial_mols = []
    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()
        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        # build_molecule already creates the mol with its initial conformer
        mol = build_molecule(pos, atom_type, dataset_info)
        initial_mols.append(mol)

    # Filter for molecules that were successfully built
    valid_initial_mols = [mol for mol in initial_mols if mol is not None]
    if not valid_initial_mols:
        print("No valid initial molecules were built.")
        return None

    # 2. Optimize each valid molecule
    # Create pairs of (initial_mol, optimized_mol)
    opt_pairs = []
    for mol in valid_initial_mols:
        optimized_mol = optimise_mol(mol)
        # Keep the pair only if optimization was successful
        if optimized_mol is not None:
            opt_pairs.append((mol, optimized_mol))
    
    if not opt_pairs:
        print("No molecules were successfully optimized.")
        return None

    # 3. Calculate energies and compute strain energy for valid pairs
    strain_energies = []
    for initial_mol, optimized_mol in opt_pairs:
        # Calculate energy for the initial pose
        initial_energy = calc_energy(initial_mol, per_atom=per_atom)
        
        # Calculate energy for the optimized pose
        optimized_energy = calc_energy(optimized_mol, per_atom=per_atom)

        # A pair is valid only if both energies can be calculated
        if _is_valid_float(initial_energy) and _is_valid_float(optimized_energy):
            strain_energy = initial_energy - optimized_energy
            strain_energies.append(strain_energy)

    if not strain_energies:
        print("Could not calculate strain energy for any valid molecule pair.")
        return None

    # 4. Compute the average strain energy
    average_strain_energy = sum(strain_energies) / len(strain_energies)
    print(f"Average Strain Energy: {average_strain_energy:.4f} over {len(strain_energies)} valid molecules")
    
    return average_strain_energy

if __name__ == '__main__':

    # main_analyze_qm9(remove_h=False, dataset_name='qm9')
    main_check_stability(remove_h=False)

