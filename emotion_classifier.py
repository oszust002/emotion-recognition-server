import math
import numpy as np
from joblib import load
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
from statsmodels.robust.scale import mad


class EmotionClassifier:

    def __init__(self, model_path, preprocessor_path=None):
        if preprocessor_path is None:
            self.preprocessor = None
        else:
            self.preprocessor = load(preprocessor_path)
        self.classifier = load(model_path)

    def classify(self, rr, hrdata=None):
        attrs, r = self.transform_into_feature_row(rr, hrdata)
        row = [r]
        scaled = row
        if self.preprocessor is not None:
            scaled = self.preprocessor.transform(scaled)
        return self.classifier.predict(scaled)[0], attrs

    def transform_into_feature_row(self, rr, hrdata=None):
        hr = hrdata
        if hr is None:
            hr = 60 / rr
        hrv_attrs = self.calculate_hrv_attrs(rr)
        mean_hr = np.mean(hr)
        mean_rr = np.mean(rr)
        std_hr = np.std(hr)
        std_rr = hrv_attrs['sdnn']

        hr_above_mean_plus_std = hr[np.where(hr > mean_hr + std_hr)]
        hr_below_mean_minus_std = hr[np.where(hr < mean_hr - std_hr)]
        rr_above_mean_plus_std = rr[np.where(rr > mean_rr + std_rr)]
        rr_below_mean_minus_std = rr[np.where(rr < mean_rr - std_rr)]

        attrs = {
            'mean_hr': mean_hr,
            'min_hr': hr.min(),
            'max_hr': hr.max(),
            'std_hr': std_hr,
            'kurtosis_hr': kurtosis(hr),
            'skewness_hr': skew(hr),
            'hr_above_mean_plus_std': len(hr_above_mean_plus_std) / len(hr),
            'hr_below_mean_minus_std': len(hr_below_mean_minus_std) / len(hr),
            'mad_rr': mad(rr),
            'mean_rr': mean_rr,
            'kurtosis_rr': kurtosis(rr),
            'skewness_rr': skew(rr),
            'rr_above_mean_plus_std': len(rr_above_mean_plus_std) / len(rr),
            'rr_below_mean_minus_std': len(rr_below_mean_minus_std) / len(rr),
            'sdnn': hrv_attrs['sdnn'],
            'rmssd': hrv_attrs['rmssd'],
            'sdsd': hrv_attrs['sdsd'],
            'pnn20': hrv_attrs['pnn20'],
            'pnn50': hrv_attrs['pnn50'],
            'sd1': hrv_attrs['sd1'],
            'sd2': hrv_attrs['sd2'],
            'sd2_sd1_ratio': hrv_attrs['sd2_sd1_ratio'],
            'lf': hrv_attrs['lf'],
            'hf': hrv_attrs['hf'],
            'lfhf': hrv_attrs['lfhf']
        }

        attrs_as_row = [attrs['mean_hr'], attrs['min_hr'], attrs['max_hr'], attrs['std_hr'], attrs['kurtosis_hr'],
                        attrs['skewness_hr'], attrs['hr_above_mean_plus_std'], attrs['hr_below_mean_minus_std'],
                        attrs['mad_rr'], attrs['mean_rr'], attrs['kurtosis_rr'], attrs['skewness_rr'],
                        attrs['rr_above_mean_plus_std'], attrs['rr_below_mean_minus_std'], attrs['sdnn'],
                        attrs['rmssd'], attrs['sdsd'], attrs['pnn20'], attrs['pnn50'], attrs['sd1'], attrs['sd2'],
                        attrs['sd2_sd1_ratio'], attrs['lf'], attrs['hf'], attrs['lfhf']]

        return attrs, attrs_as_row

    def extract_frequent_domain_features(self, rr):
        rr_in_ms = rr * 1000
        nni_tmstp = np.cumsum(rr)
        nni_tmstp = nni_tmstp - nni_tmstp[0]
        funct = interp1d(x=nni_tmstp, y=rr_in_ms, kind='cubic')
        timestamps_interpolation = np.arange(0, nni_tmstp[-1], 1 / float(4))
        nni_interpolation = funct(timestamps_interpolation)
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)
        freq, psd = signal.welch(x=nni_normalized, fs=4)
        lf_indexes = np.logical_and(freq >= 0.04, freq < 0.15)
        hf_indexes = np.logical_and(freq >= 0.15, freq < 0.4)
        lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
        hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])
        return lf, hf

    def calculate_hrv_attrs(self, rr):
        rmssd = 0
        diffs = []
        nn20 = 0
        nn50 = 0
        for i in range(1, len(rr) - 1):
            diff = abs(rr[i + 1] - rr[i])
            if diff > 0.05:
                nn50 += 1
            if diff > 0.02:
                nn20 += 1
            diffs.append(diff)
            rmssd += diff ** 2
        rmssd = math.sqrt(rmssd / (len(rr) - 1))
        sdsd = np.std(diffs)
        sd1 = np.sqrt(np.std(diffs) ** 2 * 0.5)
        sd2 = np.sqrt(2 * np.std(rr) ** 2 - 0.5 * np.std(diffs) ** 2)

        lf, hf = self.extract_frequent_domain_features(rr)

        return {
            'rmssd': rmssd,
            'pnn20': nn20 / len(rr),
            'pnn50': nn50 / len(rr),
            'sdsd': sdsd,
            'sdnn': np.std(rr),
            'sd1': sd1,
            'sd2': sd2,
            'sd2_sd1_ratio': sd2 / sd1,
            'lf': lf,
            'hf': hf,
            'lfhf': lf / hf
        }

    def load_classifier(self):
        return load('best_model.joblib')
