import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Daniel Eugenio González Limas
# A01285898


def montecarlo(n, a, b, f):

    # Generar n números aleatorios uniformes en [a, b]
    xi = np.random.uniform(a, b, n)

    # Selección de la función a evaluar
    if f == "a":
        fx = 1 / (np.exp(xi) + np.exp(-xi))
    else:
        fx = 2 / (np.exp(xi) + np.exp(-xi))

    # Cálculo de áreas individuales
    areas = fx * ((b - a) / n)

    # Estimación de la integral
    integral = np.sum(areas)

    return xi, fx, areas, integral


# Interfaz de usuario
st.set_page_config(page_title="Daniel Eugenio Gonzalez Limas ", layout="wide")
st.title("Método de Monte Carlo para Integración")
st.header("Daniel Eugenio González Limas")
st.header("A01285898")

st.header("Parámetros de Entrada")

# Selección de función a evaluar
funcion = st.radio("Selecciona la función:", ("a", "b"))

# Selección del número de muestras
n = st.slider("Número de muestras:", 100, 10000, 1000, step=100)

# Selección del intervalo a integrar
intervalo = st.radio(
    "Selecciona intervalo [a, b]:", ("[0, π/2]", "[-6, 6]", "[-∞, ∞]", "Personalizado")
)

if intervalo == "[0, π/2]":
    a, b = 0, np.pi / 2
elif intervalo == "[-6, 6]":
    a, b = -6, 6
elif intervalo == "[-∞, ∞]":
    a, b = -5000, 5000  # Números muy grandes para aproximar a infinito.
else:
    col1, col2 = st.columns(2)
    a = col1.number_input("Límite a:", value=0.0)
    b = col2.number_input("Límite b:", value=1.0)

    if b <= a:
        st.error(
            "El límite b debe ser mayor que el límite a. Ingresa nuevamente los valores."
        )

# Ejecución del método
if st.button("Calcular Integral") & (b > a):
    st.subheader("Resultados de la Simulación")

    # Ejecutar el método
    xi, fx, areas, integral = montecarlo(n, a, b, funcion)

    # Mostrar Parámetros de Entrada y Salida
    st.subheader("Parámetros de Entrada")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Función", f"f(x) = {funcion}")
    col2.metric("Número de muestras", f"{n:,}")
    col3.metric("Límite inferior (a)", f"{a:.4f}")
    col4.metric("Límite superior (b)", f"{b:.4f}")

    st.subheader("Parámetros de Salida")
    col1, col2 = st.columns(2)
    col1.metric("Estimación de la Integral", f"{integral:.6f}", delta=None)
    col2.metric("Ancho del intervalo (b-a)", f"{b-a:.6f}")

    # Mostrar los valores generados de xi, f(xi), y áreas
    st.subheader("Valores Aleatorios Generados (50 filas)")
    datos = pd.DataFrame(
        {
            "xi (valores aleatorios)": xi[:50],
            "f(xi) (altura)": fx[:50],
            "Área individual": areas[:50],
        }
    )
    st.dataframe(datos, use_container_width=True)

    st.write(f"**Total de muestras generadas:** {len(xi)}")
    st.write(f"**Suma de todas las áreas:** {np.sum(areas):.6f}")
