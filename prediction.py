import pandas as pd
import scipy.stats as st

def predict(df: pd.DataFrame, col: str, threshold: float):
    data = df[col]

    # Fit parameters for selected distributions
    norm_params = st.norm.fit(data)
    expon_params = st.expon.fit(data)
    uniform_params = st.uniform.fit(data)
    lognorm_params = st.lognorm.fit(data)

    # Calculate p-values for each distribution
    p_values = {
        'norm': st.kstest(data, 'norm', args=norm_params)[1],
        'expon': st.kstest(data, 'expon', args=expon_params)[1],
        'uniform': st.kstest(data, 'uniform', args=uniform_params)[1],
        'lognorm': st.kstest(data, 'lognorm', args=lognorm_params)[1]
    }

    best_dist = max(p_values, key=p_values.get)
    best_params = None

    # Fit parameters and calculate probability based on the selected distribution
    if best_dist == 'norm':
        best_params = st.norm.fit(data)
        prob = 1 - st.norm.cdf(threshold, *best_params)
    elif best_dist == 'expon':
        best_params = st.expon.fit(data)
        prob = 1 - st.expon.cdf(threshold, *best_params)
    elif best_dist == 'uniform':
        best_params = st.uniform.fit(data)
        prob = 1 - st.uniform.cdf(threshold, *best_params)
    else:
        best_params = st.lognorm.fit(data)
        prob = 1 - st.lognorm.cdf(threshold, *best_params)

    return prob, best_dist