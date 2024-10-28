import pandas as pd
import itertools
import streamlit as st
from datetime import datetime
import io

# Set Streamlit page config
st.set_page_config(
    page_title="Sum Value Calculator",
    page_icon="🧮",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def find_combinations(target_sum, products, values, max_items=None):
    """
    ค้นหาชุดสินค้าที่มียอดรวมเท่ากับเป้าหมาย
    parameters:
        target_sum: ยอดรวมที่ต้องการ
        products: รายชื่อสินค้า
        values: ราคาสินค้า
        max_items: จำนวนสินค้าสูงสุดในแต่ละชุด (optional)
    """
    results = []
    # กำหนดช่วงการค้นหาตาม max_items
    range_end = len(values) + 1 if max_items is None else min(max_items + 1, len(values) + 1)
    
    for r in range(1, range_end):
        for combo_indices in itertools.combinations(range(len(values)), r):
            combo = [values[i] for i in combo_indices]
            if sum(combo) == target_sum:
                product_names = [products[i] for i in combo_indices]
                results.append((product_names, combo))
    return results

def export_results(results, target):
    """สร้าง DataFrame สำหรับการ export ผลลัพธ์"""
    export_data = []
    for products, values in results:
        export_data.append({
            'เป้าหมาย': target,
            'รายการสินค้า': ', '.join(products),
            'ราคาแต่ละชิ้น': ', '.join([str(v) for v in values]),
            'จำนวนสินค้า': len(products)
        })
    return pd.DataFrame(export_data)

# หน้าหลักของแอพพลิเคชัน
st.title("🧮 เครื่องมือคำนวณยอดรวมสินค้า")
st.markdown("---")

# สร้าง sidebar สำหรับการตั้งค่า
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    max_items = st.number_input(
        "จำนวนสินค้าสูงสุดต่อชุด",
        min_value=2,
        max_value=500,
        value=5,
        help="จำกัดจำนวนสินค้าสูงสุดในแต่ละชุดเพื่อลดเวลาในการคำนวณ"
    )
    
    show_instructions = st.checkbox("แสดงคำแนะนำการใช้งาน", value=True)

if show_instructions:
    st.info("""
    📝 **วิธีใช้งาน:**
    1. อัปโหลดไฟล์ Excel ที่มีคอลัมน์ดังนี้:
        - product: ชื่อสินค้า
        - sell_value: ราคาขาย
        - netvalue: ยอดรวมที่ต้องการค้นหา
    2. ระบบจะค้นหาชุดสินค้าที่มียอดรวมเท่ากับ netvalue ที่ต้องการ
    3. สามารถดาวน์โหลดผลลัพธ์เป็นไฟล์ Excel ได้
    """)

# อัปโหลดไฟล์
uploaded_file = st.file_uploader("📎 เลือกไฟล์ Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # อ่านข้อมูลจากไฟล์ Excel
        df = pd.read_excel(uploaded_file)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['product', 'sell_value', 'netvalue']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
        else:
            # แสดงตัวอย่างข้อมูล
            st.subheader("📊 ตัวอย่างข้อมูล")
            st.dataframe(df.head())
            
            # ประมวลผล
            products = df['product'].fillna("").tolist()
            sell_values = df['sell_value'].dropna().tolist()
            net_values = df['netvalue'].dropna().tolist()
            
            if not products or not sell_values or not net_values:
                st.error("❌ ข้อมูลในคอลัมน์ไม่ถูกต้องหรือเป็นค่าว่าง")
            else:
                # สร้าง container สำหรับเก็บผลลัพธ์ทั้งหมด
                all_results = []
                
                # แสดงผลลัพธ์
                st.subheader("🔍 ผลการค้นหา")
                for target in net_values:
                    combinations = find_combinations(target, products, sell_values, max_items)
                    
                    # สร้าง expander สำหรับแต่ละเป้าหมาย
                    with st.expander(f"💰 ชุดสินค้าที่รวมกันได้ {target:,.2f} บาท"):
                        if combinations:
                            for product_names, combo in combinations:
                                st.markdown(f"""
                                🛍️ **สินค้า:** {', '.join(product_names)}  
                                💵 **ราคา:** {', '.join([f'{v:,.2f}' for v in combo])} บาท  
                                📦 **จำนวนสินค้า:** {len(combo)} ชิ้น
                                ---
                                """)
                            # เก็บผลลัพธ์สำหรับ export
                            all_results.extend([(target, c) for c in combinations])
                        else:
                            st.warning("⚠️ ไม่พบชุดสินค้าที่รวมกันได้ตามเป้าหมาย")
                
                # สร้างปุ่มดาวน์โหลดผลลัพธ์
                if all_results:
                    st.markdown("---")
                    st.subheader("📥 ดาวน์โหลดผลลัพธ์")
                    
                    # สร้าง DataFrame สำหรับ export
                    export_df = pd.DataFrame([
                        {
                            'เป้าหมาย': target,
                            'รายการสินค้า': ', '.join(products),
                            'ราคาแต่ละชิ้น': ', '.join([f'{v:,.2f}' for v in values]),
                            'จำนวนสินค้า': len(products)
                        }
                        for target, (products, values) in all_results
                    ])
                    
                    # สร้างชื่อไฟล์ที่มีวันที่และเวลา
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sum_value_results_{timestamp}.xlsx"
                    
                    # สร้าง Excel file ในหน่วยความจำ
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False)
                    
                    # เตรียมไฟล์สำหรับดาวน์โหลด
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                        data=buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.error("กรุณาตรวจสอบรูปแบบไฟล์และข้อมูลให้ถูกต้อง")