import pandas as pd
import itertools
import streamlit as st
from datetime import datetime
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cProfile
import time

# Set Streamlit page config
st.set_page_config(
    page_title="Sum Value Calculator",
    page_icon="🧮",
    layout="wide"
)

# Custom CSS with improved styling
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .filter-section {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def process_chunk(args):
    """แบ่งการประมวลผลเป็นส่วนๆ สำหรับ parallel processing"""
    chunk_indices, target_sum, values, threshold = args
    results = []
    chunk_sum = sum(values[i] for i in chunk_indices)
    if abs(chunk_sum - target_sum) <= threshold:
        product_indices = list(chunk_indices)
        results.append(product_indices)
    return results

def find_combinations_parallel(target_sum, products, values, max_items=None, threshold=0.01):
    """ค้นหาชุดสินค้าแบบ parallel processing"""
    results = []
    range_end = len(values) + 1 if max_items is None else min(max_items + 1, len(values) + 1)
    
    # ใช้ numpy เพื่อเพิ่มประสิทธิภาพการคำนวณ
    values_array = np.array(values)
    
    with ThreadPoolExecutor() as executor:
        for r in range(1, range_end):
            chunks = list(itertools.combinations(range(len(values)), r))
            chunk_size = 1000  # ปรับขนาด chunk ตามความเหมาะสม
            for i in range(0, len(chunks), chunk_size):
                chunk_batch = chunks[i:i + chunk_size]
                args = [(chunk, target_sum, values_array, threshold) for chunk in chunk_batch]
                for result in executor.map(process_chunk, args):
                    if result:
                        for indices in result:
                            combo = [values[i] for i in indices]
                            product_names = [products[i] for i in indices]
                            results.append((product_names, combo))
    return results

def export_results(results, target, include_metadata=True):
    """สร้าง DataFrame สำหรับการ export ผลลัพธ์พร้อมข้อมูลเพิ่มเติม"""
    export_data = []
    for products, values in results:
        data = {
            'เป้าหมาย': target,
            'รายการสินค้า': ', '.join(products),
            'ราคาแต่ละชิ้น': ', '.join([f'{v:,.2f}' for v in values]),
            'จำนวนสินค้า': len(products)
        }
        if include_metadata:
            data.update({
                'ยอดรวม': sum(values),
                'ราคาเฉลี่ยต่อชิ้น': sum(values) / len(values),
                'วันที่คำนวณ': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        export_data.append(data)
    return pd.DataFrame(export_data)

# เพิ่มฟังก์ชันสำหรับการวิเคราะห์ประสิทธิภาพ
def profile_execution(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        start_time = time.time()
        result = profiler.runcall(func, *args, **kwargs)
        execution_time = time.time() - start_time
        st.sidebar.markdown(f"⏱️ เวลาที่ใช้: {execution_time:.2f} วินาที")
        return result
    return wrapper

# หน้าหลักของแอพพลิเคชัน
st.title("🧮 เครื่องมือคำนวณยอดรวมสินค้า")
st.markdown("---")

# สร้าง sidebar สำหรับการตั้งค่าที่มีตัวเลือกมากขึ้น
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    
    max_items = st.number_input(
        "จำนวนสินค้าสูงสุดต่อชุด",
        min_value=2,
        max_value=500,
        value=5
    )
    
    threshold = st.slider(
        "ค่าความคลาดเคลื่อนที่ยอมรับได้ (บาท)",
        min_value=0.0,
        max_value=10.0,
        value=0.01,
        step=0.01
    )
    
    include_metadata = st.checkbox("รวมข้อมูลเพิ่มเติมในผลลัพธ์", value=True)
    show_instructions = st.checkbox("แสดงคำแนะนำการใช้งาน", value=True)
    enable_profiling = st.checkbox("เปิดการวิเคราะห์ประสิทธิภาพ", value=False)

# แสดงคำแนะนำการใช้งานที่ละเอียดขึ้น
if show_instructions:
    st.info("""
    📝 **วิธีใช้งาน:**
    1. **การเตรียมไฟล์:**
        - สร้างไฟล์ Excel ที่มีคอลัมน์ product, sell_value, และ netvalue
        - ตรวจสอบว่าข้อมูลไม่มีค่าว่างหรือข้อมูลที่ไม่ถูกต้อง
    
    2. **การตั้งค่า:**
        - ปรับจำนวนสินค้าสูงสุดต่อชุดเพื่อควบคุมเวลาการคำนวณ
        - กำหนดค่าความคลาดเคลื่อนที่ยอมรับได้
        - เลือกรวมข้อมูลเพิ่มเติมในผลลัพธ์
    
    3. **การใช้งานขั้นสูง:**
        - เปิดการวิเคราะห์ประสิทธิภาพเพื่อดูเวลาที่ใช้ในการคำนวณ
        - ใช้ตัวกรองเพื่อค้นหาผลลัพธ์ที่ต้องการ
    """)

# อัปโหลดไฟล์
uploaded_file = st.file_uploader("📎 เลือกไฟล์ Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # อ่านข้อมูลจากไฟล์ Excel
        df = pd.read_excel(uploaded_file)
        
        # ตรวจสอบและทำความสะอาดข้อมูล
        required_columns = ['product', 'sell_value', 'netvalue']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ ไม่พบคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")
        else:
            # แสดงข้อมูลสถิติเบื้องต้น
            st.subheader("📊 ข้อมูลเบื้องต้น")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("จำนวนสินค้าทั้งหมด", len(df))
            with col2:
                st.metric("ราคาเฉลี่ย", f"{df['sell_value'].mean():,.2f}")
            with col3:
                st.metric("จำนวนเป้าหมาย", len(df['netvalue'].unique()))
            
            # แสดงตัวอย่างข้อมูล
            st.dataframe(df.head())
            
            # เพิ่มตัวกรองข้อมูล
            st.subheader("🔍 ตัวกรองข้อมูล")
            col1, col2 = st.columns(2)
            with col1:
                min_price = st.number_input("ราคาต่ำสุด", value=float(df['sell_value'].min()))
            with col2:
                max_price = st.number_input("ราคาสูงสุด", value=float(df['sell_value'].max()))
            
            # กรองข้อมูล
            df_filtered = df[
                (df['sell_value'] >= min_price) &
                (df['sell_value'] <= max_price)
            ]
            
            # ประมวลผลด้วยฟังก์ชันที่ปรับปรุงแล้ว
            products = df_filtered['product'].fillna("").tolist()
            sell_values = df_filtered['sell_value'].dropna().tolist()
            net_values = df_filtered['netvalue'].dropna().unique().tolist()
            
            if not products or not sell_values or not net_values:
                st.error("❌ ข้อมูลในคอลัมน์ไม่ถูกต้องหรือเป็นค่าว่าง")
            else:
                # สร้าง progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ประมวลผลและแสดงผลลัพธ์
                all_results = []
                for i, target in enumerate(net_values):
                    status_text.text(f"กำลังค้นหาชุดสินค้าสำหรับเป้าหมาย {target:,.2f} บาท...")
                    
                    if enable_profiling:
                        combinations = profile_execution(find_combinations_parallel)(
                            target, products, sell_values, max_items, threshold
                        )
                    else:
                        combinations = find_combinations_parallel(
                            target, products, sell_values, max_items, threshold
                        )
                    
                    if combinations:
                        with st.expander(f"💰 ชุดสินค้าที่รวมกันได้ {target:,.2f} บาท"):
                            for product_names, combo in combinations:
                                st.markdown(f"""
                                🛍️ **สินค้า:** {', '.join(product_names)}  
                                💵 **ราคา:** {', '.join([f'{v:,.2f}' for v in combo])} บาท  
                                📦 **จำนวนสินค้า:** {len(combo)} ชิ้น
                                💰 **ยอดรวม:** {sum(combo):,.2f} บาท
                                ---
                                """)
                        all_results.extend([(target, c) for c in combinations])
                    
                    progress_bar.progress((i + 1) / len(net_values))
                
                status_text.text("✅ การค้นหาเสร็จสิ้น")
                progress_bar.empty()
                
                # สร้างปุ่มดาวน์โหลดผลลัพธ์
                if all_results:
                    st.markdown("---")
                    st.subheader("📥 ดาวน์โหลดผลลัพธ์")
                    
                    # สร้าง DataFrame สำหรับ export
                    export_df = export_results(
                        all_results,
                        [target for target, _ in all_results],
                        include_metadata
                    )
                    
                    # สร้างชื่อไฟล์
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sum_value_results_{timestamp}.xlsx"
                    
                    # สร้าง Excel file
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False)
                    
                    # ปุ่มดาวน์โหลด
                    st.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                        data=buffer.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # แสดงสรุปผลการค้นหา
                    st.subheader("📊 สรุปผลการค้นหา")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("จำนวนชุดที่พบ", len(all_results))
                    with col2:
                        avg_items = np.mean([len(products) for _, (products, _) in all_results])
                        st.metric("จำนวนสินค้าเฉลี่ยต่อชุด", f"{avg_items:.1f}")
                    with col3:
                        success_rate = (len(all_results) / len(net_values)) * 100
                        st.metric("อัตราการค้นพบ", f"{success_rate:.1f}%")

                    # แสดงกราฟวิเคราะห์
                    st.subheader("📈 การวิเคราะห์ผลลัพธ์")
                    
                    # สร้าง DataFrame สำหรับการวิเคราะห์
                    analysis_df = pd.DataFrame([
                        {
                            'เป้าหมาย': target,
                            'จำนวนสินค้า': len(products),
                            'ยอดรวม': sum(values)
                        }
                        for target, (products, values) in all_results
                    ])

                    # แสดงการกระจายของจำนวนสินค้าต่อชุด
                    col1, col2 = st.columns(2)
                    with col1:
                        items_dist = analysis_df['จำนวนสินค้า'].value_counts().sort_index()
                        st.bar_chart(items_dist, use_container_width=True)
                        st.caption("การกระจายของจำนวนสินค้าต่อชุด")

                    with col2:
                        # แสดงความสัมพันธ์ระหว่างเป้าหมายและจำนวนสินค้า
                        st.scatter_chart(
                            data=analysis_df,
                            x='เป้าหมาย',
                            y='จำนวนสินค้า',
                            use_container_width=True
                        )
                        st.caption("ความสัมพันธ์ระหว่างเป้าหมายและจำนวนสินค้า")

                    # แสดงตารางสรุปผล
                    st.subheader("📋 ตารางสรุปผล")
                    summary_table = analysis_df.describe()
                    st.dataframe(summary_table)

                    # เพิ่มตัวเลือกการกรองผลลัพธ์
                    st.subheader("🔍 กรองผลลัพธ์")
                    col1, col2 = st.columns(2)
                    with col1:
                        min_items = st.number_input(
                            "จำนวนสินค้าต่ำสุด",
                            min_value=int(analysis_df['จำนวนสินค้า'].min()),
                            max_value=int(analysis_df['จำนวนสินค้า'].max()),
                            value=int(analysis_df['จำนวนสินค้า'].min())
                        )
                    with col2:
                        max_items = st.number_input(
                            "จำนวนสินค้าสูงสุด",
                            min_value=int(analysis_df['จำนวนสินค้า'].min()),
                            max_value=int(analysis_df['จำนวนสินค้า'].max()),
                            value=int(analysis_df['จำนวนสินค้า'].max())
                        )

                    # กรองและแสดงผลลัพธ์ที่กรองแล้ว
                    filtered_results = analysis_df[
                        (analysis_df['จำนวนสินค้า'] >= min_items) &
                        (analysis_df['จำนวนสินค้า'] <= max_items)
                    ]
                    st.dataframe(filtered_results)

                    # เพิ่มปุ่มดาวน์โหลดผลลัพธ์ที่กรองแล้ว
                    if not filtered_results.empty:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            filtered_results.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์ที่กรอง (Excel)",
                            data=buffer.getvalue(),
                            file_name=f"filtered_results_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                else:
                    st.warning("⚠️ ไม่พบชุดสินค้าที่ตรงตามเงื่อนไข")
                    
                # แสดงคำแนะนำเพิ่มเติม
                st.markdown("""
                💡 **คำแนะนำ:**
                - หากไม่พบผลลัพธ์ ลองปรับค่าความคลาดเคลื่อนที่ยอมรับได้ให้สูงขึ้น
                - ลองปรับจำนวนสินค้าสูงสุดต่อชุดเพื่อค้นหาผลลัพธ์เพิ่มเติม
                - ใช้ตัวกรองเพื่อค้นหาชุดสินค้าที่ตรงตามความต้องการ
                """)
                
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.error("กรุณาตรวจสอบรูปแบบไฟล์และข้อมูลให้ถูกต้อง")
        st.markdown("""
        🔍 **สาเหตุที่เป็นไปได้:**
        1. รูปแบบไฟล์ไม่ถูกต้อง
        2. ข้อมูลในคอลัมน์ไม่ตรงตามที่กำหนด
        3. มีค่าว่างหรือข้อมูลที่ไม่ถูกต้องในไฟล์
        
        📝 **วิธีแก้ไข:**
        1. ตรวจสอบว่าไฟล์เป็น Excel (.xlsx หรือ .xls)
        2. ตรวจสอบชื่อคอลัมน์ให้ถูกต้อง
        3. ทำความสะอาดข้อมูลก่อนอัปโหลด
        """)