import pandas as pd
import itertools
import streamlit as st

# ฟังก์ชันหาชุดตัวเลขที่รวมกันได้ตามเป้าหมาย พร้อมแสดงชื่อสินค้า
def find_combinations(target_sum, products, values):
    results = []
    for r in range(1, len(values) + 1):
        for combo_indices in itertools.combinations(range(len(values)), r):
            combo = [values[i] for i in combo_indices]
            if sum(combo) == target_sum:
                product_names = [products[i] for i in combo_indices]
                results.append((product_names, combo))
    return results

# ส่วนหลักของโปรแกรม
st.title("อัปโหลดไฟล์ Excel และคำนวณยอดรวม")

# อัปโหลดไฟล์ Excel
uploaded_file = st.file_uploader("เลือกไฟล์ Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        products = df['product'].fillna("").tolist()
        sell_values = df['sell_value'].dropna().tolist()
        net_values = df['netvalue'].dropna().tolist()

        if not products or not sell_values or not net_values:
            st.error("ข้อมูลในคอลัมน์ไม่ถูกต้อง")
        else:
            for target in net_values:
                combinations = find_combinations(target, products, sell_values)
                st.write(f"### ชุดตัวเลขที่รวมกันได้ {target}:")
                if combinations:
                    for product_names, combo in combinations:
                        st.write(f"สินค้า: {product_names} | ยอด: {combo}")
                else:
                    st.write("ไม่พบชุดตัวเลขที่รวมกันได้")
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")
