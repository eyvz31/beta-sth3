import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

KB = 1.380649e-23
NA = 6.02214076e23
PI = math.pi

MATERIALS = {
    "Generic hydrogel": {"permeability": 1.00, "binding": 0.05},
    "PEG-like hydrogel": {"permeability": 1.10, "binding": 0.02},
    "Alginate-like hydrogel": {"permeability": 0.90, "binding": 0.04},
    "Polymer membrane": {"permeability": 0.75, "binding": 0.08},
    "Custom": {"permeability": 1.00, "binding": 0.05},
}


class Molecule:
    def __init__(
        self,
        name,
        molecular_weight,
        hydrodynamic_radius_nm,
        diffusion_coefficient_m2s=0.0,
        shape_factor=1.0,
        binding_fraction=0.0,
    ):
        self.name = name
        self.mw = molecular_weight
        self.radius_nm = hydrodynamic_radius_nm
        self.D0 = diffusion_coefficient_m2s
        self.shape = shape_factor
        self.binding = binding_fraction


def stokes_einstein(radius_nm, temperature_K, viscosity=0.000691):
    radius_m = radius_nm * 1e-9
    if radius_m <= 0:
        return 0.0
    return KB * temperature_K / (6 * PI * viscosity * radius_m)


def molecular_radius_from_MW(mw_da, density=1350.0):
    mass_kg = mw_da / NA / 1000.0
    volume_m3 = mass_kg / density
    radius_m = ((3.0 * volume_m3) / (4.0 * PI)) ** (1.0 / 3.0)
    return radius_m * 1e9


def steric_partition(pore_radius_nm, molecular_radius_nm, shape_factor=1.0):
    effective_radius = molecular_radius_nm * max(shape_factor, 0.01)
    if pore_radius_nm <= effective_radius:
        return 0.0
    lam = effective_radius / pore_radius_nm
    return max(0.0, min(1.0, (1.0 - lam) ** 2))


def pore_distribution_factor(pore_diameter_nm, mean_pore_nm, std_pore_nm):
    if std_pore_nm <= 0:
        return 1.0
    x = (pore_diameter_nm - mean_pore_nm) / std_pore_nm
    return math.exp(-0.5 * x * x)


def effective_diffusion(
    molecule,
    pore_diameter_nm,
    porosity,
    tortuosity,
    membrane_material_factor,
    fouling_fraction,
):
    pore_radius = pore_diameter_nm / 2.0
    steric = steric_partition(
        pore_radius, molecule.radius_nm, molecule.shape
    )

    if molecule.D0 <= 0:
        raise ValueError(
            f"D0 for {molecule.name} must be greater than zero."
        )

    if tortuosity <= 0:
        return 0.0

    binding_penalty = 1.0 - max(
        0.0, min(0.99, molecule.binding + fouling_fraction)
    )

    return max(
        0.0,
        molecule.D0
        * steric
        * max(0.0, min(1.0, porosity))
        / (tortuosity**2)
        * membrane_material_factor
        * binding_penalty,
    )


def diffusion_time(membrane_thickness_um, D_eff):
    if D_eff <= 0:
        return float("inf")
    L = membrane_thickness_um * 1e-6
    return L * L / (2.0 * D_eff)


def transport_index(molecule, pore_diameter_nm, fouling=0.0):
    steric = steric_partition(
        pore_diameter_nm / 2.0, molecule.radius_nm, molecule.shape
    )
    binding_penalty = 1.0 - min(0.99, max(0.0, molecule.binding + fouling))
    return max(0.0, min(1.0, steric * binding_penalty))


def calculate_composite_score(g_tr, i_tr, a_tr):
    """Единая функция расчета Composite Score"""
    exclusion = 1.0 - a_tr
    return 0.35 * g_tr + 0.35 * i_tr + 0.30 * exclusion


st.set_page_config(layout="wide")


col_left, col_middle = st.columns(2)

with col_left:
    st.subheader("Molecules")

    st.markdown("**GLUCOSE**")
    glucose_mw = st.number_input("MW (Da)", value=180.16, key="g_mw")
    glucose_radius = st.number_input(
        "Hydrodynamic radius (nm)", value=0.36, key="g_r"
    )
    glucose_shape = st.number_input("Shape factor", value=1.0, key="g_s")
    binding_glucose = st.number_input(
        "Binding fraction", value=0.00, key="g_b"
    )

    st.markdown("---")
    st.markdown("**INSULIN**")
    insulin_mw = st.number_input("MW (Da)", value=5808.0, key="i_mw")
    insulin_radius = st.number_input(
        "Hydrodynamic radius (nm)", value=1.35, key="i_r"
    )
    insulin_shape = st.number_input("Shape factor", value=1.0, key="i_s")
    binding_insulin = st.number_input(
        "Binding fraction", value=0.05, key="i_b"
    )

    st.markdown("---")
    st.markdown("**ANTIBODY / IgG**")
    antibody_mw = st.number_input("MW (Da)", value=150000.0, key="a_mw")
    antibody_radius = st.number_input(
        "Hydrodynamic radius (nm)", value=5.5, key="a_r"
    )
    antibody_shape = st.number_input("Shape factor", value=1.2, key="a_s")
    binding_antibody = st.number_input(
        "Binding fraction", value=0.15, key="a_b"
    )

with col_middle:
    st.subheader("Membrane")
    material_name = st.selectbox(
        "Material", list(MATERIALS.keys()), index=0
    )

    if "pore_val" not in st.session_state:
        st.session_state.pore_val = 8.0

    pore = st.number_input(
        "Mean pore diameter (nm)",
        value=st.session_state.pore_val,
        key="pore_input",
    )
    st.session_state.pore_val = pore

    pore_std = st.number_input("Pore SD (nm)", value=1.0)
    porosity = st.number_input("Porosity (0-1)", value=0.25)
    tortuosity = st.number_input("Tortuosity", value=2.0)
    thickness = st.number_input("Thickness (µm)", value=10.0)
    fouling = st.number_input("Fouling fraction", value=0.05)
    temperature = st.number_input("Temperature (°C)", value=37.0)

    btn_calc = st.button("CALCULATE", use_container_width=True)
    btn_opt = st.button("OPTIMIZE PORE SIZE", use_container_width=True)
    btn_sens = st.button("SENSITIVITY ANALYSIS", use_container_width=True)

temperature_K = temperature + 273.15
viscosity = 0.000691

glucose_D = stokes_einstein(glucose_radius, temperature_K, viscosity)
insulin_D = stokes_einstein(insulin_radius, temperature_K, viscosity)
antibody_D = stokes_einstein(antibody_radius, temperature_K, viscosity)

molecules = [
    Molecule(
        "Glucose",
        glucose_mw,
        glucose_radius,
        glucose_D,
        glucose_shape,
        binding_glucose,
    ),
    Molecule(
        "Insulin",
        insulin_mw,
        insulin_radius,
        insulin_D,
        insulin_shape,
        binding_insulin,
    ),
    Molecule(
        "Antibody / IgG",
        antibody_mw,
        antibody_radius,
        antibody_D,
        antibody_shape,
        binding_antibody,
    ),
]

material_data = MATERIALS[material_name]

st.markdown("---")
res_col, plot_col = st.columns([1, 1])

if btn_opt:
    best = None
    pore_values = np.linspace(0.5, 30.0, 600)

    for p in pore_values:
        g_tr = transport_index(molecules[0], p, fouling)
        i_tr = transport_index(molecules[1], p, fouling)
        a_tr = transport_index(molecules[2], p, fouling)

        exclusion = 1.0 - a_tr

        if a_tr > 0.01:
            score = 0.0
        else:
            score = calculate_composite_score(g_tr, i_tr, a_tr)

        candidate = (score, p, g_tr, i_tr, exclusion)

        if best is None or candidate[0] > best[0]:
            best = candidate

    score, opt_pore, g, i, exclusion = best
    st.session_state.pore_val = round(opt_pore, 3)

    text = (
        "OPTIMIZATION RESULT\n\n"
        f"Mean pore diameter: {opt_pore:.3f} nm\n"
        f"Glucose transport index: {g*100:.3f}%\n"
        f"Insulin transport index: {i*100:.3f}%\n"
        f"Antibody exclusion: {exclusion*100:.3f}%\n"
        f"Composite score: {score*100:.3f}%\n"
    )

    with res_col:
        st.subheader("Results")
        st.text(text)

    with plot_col:
        st.subheader("Plots")
        fig, ax = plt.subplots(figsize=(7, 5))
        p_vals = np.linspace(0.5, max(30.0, opt_pore * 3), 300)
        for molecule in molecules:
            vals = [
                transport_index(molecule, p, fouling) * 100 for p in p_vals
            ]
            ax.plot(p_vals, vals, label=molecule.name)
        ax.axvline(opt_pore, linestyle="--", color="gray", label="Optimal Pore")
        ax.set_xlabel("Pore diameter (nm)")
        ax.set_ylabel("Transport index (%)")
        ax.set_title("Transport vs Pore Diameter")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

elif btn_sens:
    parameters = ["pore", "porosity", "tortuosity", "thickness", "fouling"]
    results = []

    for parameter in parameters:
        values = []
        scores = []
        for factor in np.linspace(0.5, 1.5, 21):
            p_pore = pore
            p_porosity = porosity
            p_tortuosity = tortuosity
            p_thickness = thickness
            p_fouling = fouling

            if parameter == "pore":
                p_pore *= factor
            elif parameter == "porosity":
                p_porosity = min(0.999, porosity * factor)
            elif parameter == "tortuosity":
                p_tortuosity *= factor
            elif parameter == "thickness":
                p_thickness *= factor
            elif parameter == "fouling":
                p_fouling = min(0.99, fouling * factor)

            g = transport_index(molecules[0], p_pore, p_fouling)
            i = transport_index(molecules[1], p_pore, p_fouling)
            a = transport_index(molecules[2], p_pore, p_fouling)

            score = calculate_composite_score(g, i, a)
            values.append(factor)
            scores.append(score * 100)

        results.append((parameter, values, scores))

    text = (
        "SENSITIVITY ANALYSIS\n\n"
        "Parameter range: 0.5x - 1.5x\n\n"
        "Parameters:\n"
        "Pore diameter\n"
        "Porosity\n"
        "Tortuosity\n"
        "Thickness\n"
        "Fouling\n"
    )

    with res_col:
        st.subheader("Results")
        st.text(text)

    with plot_col:
        st.subheader("Plots")
        fig, ax = plt.subplots(figsize=(7, 5))
        for param, x, y in results:
            ax.plot(x, y, label=param)
        ax.set_xlabel("Parameter multiplier")
        ax.set_ylabel("Composite score (%)")
        ax.set_title("Sensitivity Analysis")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

else:
    lines = [
        "",
        f"Material:               {material_name}",
        f"Mean pore diameter:    {pore:.3f} nm",
        f"Pore SD:               {pore_std:.3f} nm",
        f"Porosity:              {porosity:.4f}",
        f"Tortuosity:            {tortuosity:.4f}",
        f"Thickness:             {thickness:.3f} µm",
        f"Temperature:           {temperature:.2f} °C",
        f"Fouling fraction:      {fouling:.4f}",
        "",
    ]

    transports = {}

    for molecule in molecules:
        D_eff = effective_diffusion(
            molecule,
            pore,
            porosity,
            tortuosity,
            material_data["permeability"],
            fouling,
        )
        time_s = diffusion_time(thickness, D_eff)
        tr = transport_index(molecule, pore, fouling)
        transports[molecule.name] = tr

        lines.extend(
            [
                molecule.name,
                f"  MW:                    {molecule.mw:.3f} Da",
                f"  Radius:                {molecule.radius_nm:.4f} nm",
                f"  Shape factor:          {molecule.shape:.4f}",
                f"  Free D:                {molecule.D0:.4e} m²/s",
                f"  Effective D:           {D_eff:.4e} m²/s",
                (
                    "  Characteristic time:   infinite"
                    if math.isinf(time_s)
                    else f"  Characteristic time:   {time_s/60:.4f} min"
                ),
                f"  Transport index:       {tr*100:.2f}%",
                "",
            ]
        )

    antibody_transport = transports["Antibody / IgG"]
    glucose_transport = transports["Glucose"]
    insulin_transport = transports["Insulin"]

    score = calculate_composite_score(
        glucose_transport, insulin_transport, antibody_transport
    )

    lines.extend(
        [
            "SYSTEM INDICES",
            "",
            f"Glucose transport:      {glucose_transport*100:.2f}%",
            f"Insulin transport:      {insulin_transport*100:.2f}%",
            f"Antibody exclusion:     {(1.0 - antibody_transport)*100:.2f}%",
            f"Composite score:        {score*100:.2f}%",
            "",
        ]
    )

    with res_col:
        st.subheader("Results")
        st.text("\n".join(lines))

    with plot_col:
        st.subheader("Plots")
        fig, ax = plt.subplots(figsize=(7, 5))
        pore_values = np.linspace(0.5, max(30.0, pore * 3), 300)

        for molecule in molecules:
            values = [
                transport_index(molecule, p, fouling) * 100 for p in pore_values
            ]
            ax.plot(pore_values, values, label=molecule.name)

        ax.axvline(pore, linestyle="--", color="gray", label="Current Pore")
        ax.set_xlabel("Pore diameter (nm)")
        ax.set_ylabel("Transport index (%)")
        ax.set_title("Transport vs Pore Diameter")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
