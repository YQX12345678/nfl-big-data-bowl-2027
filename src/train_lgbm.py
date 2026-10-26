import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_wr_models(X_train, X_test, y_train, y_test, base_feats):
    model_base = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=7, random_state=42, verbose=-1)
    model_base.fit(X_train[base_feats], y_train)
    mae_base = mean_absolute_error(y_test, model_base.predict(X_test[base_feats]))
    r2_base = r2_score(y_test, model_base.predict(X_test[base_feats]))
    
    model_aug = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=7, random_state=42, verbose=-1)
    model_aug.fit(X_train, y_train)
    mae_aug = mean_absolute_error(y_test, model_aug.predict(X_test))
    r2_aug = r2_score(y_test, model_aug.predict(X_test))
    
    print(f"基线模型 MAE: {mae_base:.4f} | R2: {r2_base:.4f}")
    print(f"增强模型 MAE: {mae_aug:.4f} | R2: {r2_aug:.4f}")
    print(f"误差降低幅度: {(mae_base - mae_aug) / mae_base * 100:.2f}%")
    
    return model_aug
