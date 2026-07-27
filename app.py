import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/car_price_pipeline.pkl")

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🚗 Car Price Prediction")

    st.markdown("---")

    st.subheader("📊 Model")
    st.write("Linear Regression")

    st.subheader("📁 Dataset")
    st.write("205 Cars")

    st.subheader("🎯 Model Accuracy")
    st.metric("R² Score", "0.8725")

    st.markdown("---")

    st.subheader("🛠 Built With")
    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-Learn")
    st.write("Streamlit")

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Car Price Prediction")

st.write(
    "Enter the vehicle specifications below and click **Predict Price**."
)

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        symboling = st.slider(
            "Symboling",
            -2,
            3,
            0
        )

        fueltype = st.selectbox(
            "Fuel Type",
            ["gas", "diesel"]
        )

        aspiration = st.selectbox(
            "Aspiration",
            ["std", "turbo"]
        )

        doornumber = st.selectbox(
            "Doors",
            ["two", "four"]
        )

        carbody = st.selectbox(
            "Car Body",
            [
                "convertible",
                "hardtop",
                "hatchback",
                "sedan",
                "wagon"
            ]
        )

        drivewheel = st.selectbox(
            "Drive Wheel",
            [
                "fwd",
                "rwd",
                "4wd"
            ]
        )

        enginelocation = st.selectbox(
            "Engine Location",
            [
                "front",
                "rear"
            ]
        )

        enginetype = st.selectbox(
            "Engine Type",
            [
                "dohc",
                "dohcv",
                "l",
                "ohc",
                "ohcf",
                "ohcv",
                "rotor"
            ]
        )

        cylindernumber = st.selectbox(
            "Cylinders",
            [
                "two",
                "three",
                "four",
                "five",
                "six",
                "eight",
                "twelve"
            ]
        )

    with col2:

        fuelsystem = st.selectbox(
            "Fuel System",
            [
                "1bbl",
                "2bbl",
                "4bbl",
                "idi",
                "mfi",
                "mpfi",
                "spdi",
                "spfi"
            ]
        )

        wheelbase = st.slider(
            "Wheelbase",
            86.6,
            120.9,
            95.0
        )

        carlength = st.slider(
            "Car Length",
            141.1,
            208.1,
            170.0
        )

        carwidth = st.slider(
            "Car Width",
            60.3,
            72.3,
            65.5
        )

        carheight = st.slider(
            "Car Height",
            47.8,
            59.8,
            54.0
        )

        curbweight = st.slider(
            "Curb Weight",
            1488,
            4066,
            2500
        )

        enginesize = st.slider(
            "Engine Size",
            61,
            326,
            120
        )

        boreratio = st.slider(
            "Bore Ratio",
            2.54,
            3.94,
            3.20
        )

        stroke = st.slider(
            "Stroke",
            2.07,
            4.17,
            3.20
        )

        compressionratio = st.slider(
            "Compression Ratio",
            7.0,
            23.0,
            9.0
        )

        horsepower = st.slider(
            "Horsepower",
            48,
            288,
            100
        )

        peakrpm = st.slider(
            "Peak RPM",
            4150,
            6600,
            5200
        )

        citympg = st.slider(
            "City MPG",
            13,
            49,
            25
        )

        highwaympg = st.slider(
            "Highway MPG",
            16,
            54,
            30
        )

    submitted = st.form_submit_button(
        "🚀 Predict Price"
    )
    # --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    input_df = pd.DataFrame([{

        "symboling": symboling,
        "fueltype": fueltype,
        "aspiration": aspiration,
        "doornumber": doornumber,
        "carbody": carbody,
        "drivewheel": drivewheel,
        "enginelocation": enginelocation,
        "wheelbase": wheelbase,
        "carlength": carlength,
        "carwidth": carwidth,
        "carheight": carheight,
        "curbweight": curbweight,
        "enginetype": enginetype,
        "cylindernumber": cylindernumber,
        "enginesize": enginesize,
        "fuelsystem": fuelsystem,
        "boreratio": boreratio,
        "stroke": stroke,
        "compressionratio": compressionratio,
        "horsepower": horsepower,
        "peakrpm": peakrpm,
        "citympg": citympg,
        "highwaympg": highwaympg

    }])

    prediction = model.predict(input_df)[0]

    st.markdown("---")

    st.markdown(
    f"""
<div style="
background:#0f172a;
padding:30px;
border-radius:15px;
text-align:center;
border:2px solid #2563eb;
">

<h2 style="color:white;">
🚗 Predicted Car Price
</h2>

<h1 style="color:#22c55e;font-size:55px;">
${prediction:,.2f}
</h1>

<p style="color:white;">
Estimated market value based on the selected specifications.
</p>

</div>
""",
    unsafe_allow_html=True
)

    st.markdown("")

    m1, m2, m3 = st.columns(3)

    m1.metric("Model", "Linear Regression")
    m2.metric("R² Score", "0.8725")
    m3.metric("Dataset", "205 Cars")

    st.markdown("---")

    # -----------------------------------------
    # Price Category
    # -----------------------------------------

    if prediction < 10000:
        category = "💚 Budget Car"

    elif prediction < 20000:
        category = "💙 Mid-Range Car"

    else:
        category = "❤️ Luxury Car"

    st.info(category)

    st.markdown("---")

    st.subheader("🚗 Vehicle Summary")

    left, right = st.columns(2)

    with left:

        st.write(f"**Fuel Type:** {fueltype.title()}")
        st.write(f"**Car Body:** {carbody.title()}")
        st.write(f"**Drive Wheel:** {drivewheel.upper()}")
        st.write(f"**Engine Type:** {enginetype.upper()}")
        st.write(f"**Cylinders:** {cylindernumber.title()}")
        st.write(f"**Engine Size:** {enginesize}")

    with right:

        st.write(f"**Horsepower:** {horsepower}")
        st.write(f"**Peak RPM:** {peakrpm}")
        st.write(f"**City MPG:** {citympg}")
        st.write(f"**Highway MPG:** {highwaympg}")
        st.write(f"**Fuel System:** {fuelsystem.upper()}")
        st.write(f"**Compression Ratio:** {compressionratio}")

    st.markdown("---")

    st.subheader("💡 Recommendation")

    if prediction < 10000:

        st.success("""
✔ Excellent budget-friendly option.

• Low ownership cost

• Great for city driving

• Suitable for first-time buyers
""")

    elif prediction < 20000:

        st.info("""
✔ Good value for money.

• Balanced performance

• Comfortable family vehicle

• Reliable daily use
""")

    else:

        st.warning("""
✔ Premium segment vehicle.

• Better performance

• Luxury features

• Higher maintenance cost
""")
            # -----------------------------------------
    # Model Performance Graphs
    # -----------------------------------------

    st.markdown("---")

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "images/actual_vs_predicted.png",
            caption="Actual vs Predicted Prices",
            use_container_width=True
        )

    with col2:

        st.image(
            "images/residual_plot.png",
            caption="Residual Plot",
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------------------
    # Additional Model Information
    # -----------------------------------------

    st.subheader("📊 Model Statistics")

    stat1, stat2, stat3 = st.columns(3)

    stat1.metric(
        label="MAE",
        value="$2,244"
    )

    stat2.metric(
        label="RMSE",
        value="$3,173"
    )

    stat3.metric(
        label="R² Score",
        value="0.8725"
    )

    st.markdown("---")

    # -----------------------------------------
    # Footer
    # -----------------------------------------

    st.markdown(
        """
        <div style="text-align:center;padding:20px;">
            <h4>🚗 Car Price Prediction using Machine Learning</h4>
            <p>
                Built with ❤️ using
                <b>Python</b>,
                <b>Pandas</b>,
                <b>Scikit-Learn</b>,
                <b>Streamlit</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
        