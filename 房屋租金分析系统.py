"""
房屋租金分析系统 v1.0
模块组成：
1. 数据预处理模块
2. 租金预测模型
3. 竞争力评分系统
4. Streamlit可视化模块
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ======================
# 模块1：数据预处理
# ======================
def data_preprocessing(df):
    """
    数据预处理步骤：
    1. 处理缺失值
    2. 特征工程
    3. 编码分类变量
    4. 计算衍生字段
    """
    # 缺失值处理
    df['建成年份'].fillna(df['建成年份'].median(), inplace=True)
    df['房屋年龄'] = 2024 - df['建成年份']  # 计算实际房龄
    
    # 布尔字段转换
    bool_cols = ['有阳台','有厨房','有电梯','有花园','是新建筑']
    df[bool_cols] = df[bool_cols].astype(int)
    
    # 分类变量编码
    cat_cols = ['区域1','区域2','街道','房屋类型','内饰质量']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 日期处理
    df['上传日期'] = pd.to_datetime(df['上传日期'])
    df['上传月份'] = df['上传日期'].dt.month
    
    # 删除无关字段
    df.drop(['ID','上传日期','建成年份'], axis=1, inplace=True)
    
    return df

# ======================
# 模块2：租金预测模型
# ======================
def train_rent_model(df):
    """训练随机森林回归模型"""
    X = df.drop('房屋租金', axis=1)
    y = df['房屋租金']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # 模型训练
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    
    # 保存模型和预处理对象
    joblib.dump(model, 'rent_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler

# ======================
# 模块3：竞争力评分系统
# ======================
def calculate_competitiveness(df, model, scaler):
    """计算房源竞争力得分"""
    # 生成预测租金
    X = df.drop('房屋租金', axis=1)
    X_scaled = scaler.transform(X)
    df['预测租金'] = model.predict(X_scaled)
    
    # 计算性价比指数
    df['租金差异率'] = (df['预测租金'] - df['房屋租金']) / df['预测租金']
    
    # 竞争力评分公式
    df['竞争力得分'] = (
        0.4 * df['租金差异率'] +
        0.2 * (df['居住面积'] / df['房屋租金']) +
        0.2 * df['内饰质量_编码'] +
        0.1 * df['上传图片数'] +
        0.1 * (df['有阳台'] + df['有厨房'] + df['有电梯'])
    )
    
    # 标准化得分到0-100分
    df['竞争力得分'] = (df['竞争力得分'] - df['竞争力得分'].min()
    ) / (df['竞争力得分'].max() - df['竞争力得分'].min()) * 100
    
    return df.sort_values('竞争力得分', ascending=False)

# ======================
# 模块4：Streamlit可视化
# ======================
def streamlit_dashboard(df):
    """可视化分析看板"""
    st.title("🏠 房源竞争力分析看板")
    
    # 数据筛选
    selected_area = st.sidebar.selectbox("选择区域", df['区域1'].unique())
    filtered_df = df[df['区域1'] == selected_area]
    
    # 关键指标展示
    col1, col2, col3 = st.columns(3)
    col1.metric("平均竞争力得分", f"{filtered_df['竞争力得分'].mean():.1f}")
    col2.metric("高性价比房源数", len(filtered_df[filtered_df['竞争力得分'] > 80]))
    col3.metric("平均租金差异率", f"{filtered_df['租金差异率'].mean():.2%}")
    
    # 竞争力分布地图
    st.subheader("街道竞争力分布")
    street_scores = filtered_df.groupby('街道')['竞争力得分'].mean().sort_values()
    st.bar_chart(street_scores)
    
    # 散点图分析
    st.subheader("租金与面积关系")
    st.scatter_chart(filtered_df, x='居住面积', y='房屋租金', color='竞争力得分')
    
    # 显示高性价比房源
    st.subheader("🏆 高性价比房源Top10")
    st.dataframe(filtered_df[['街道', '房屋租金', '预测租金', '竞争力得分']]
                .sort_values('竞争力得分', ascending=False).head(10))

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 加载数据（示例路径）
    df = pd.read_csv("housing_data.csv")
    
    # 执行预处理
    processed_df = data_preprocessing(df)
    
    # 训练模型（首次运行后可以注释掉）
    # model, scaler = train_rent_model(processed_df)
    
    # 加载已有模型
    model = joblib.load('rent_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # 计算竞争力
    scored_df = calculate_competitiveness(processed_df, model, scaler)
    
    # 启动可视化
    streamlit_dashboard(scored_df)
