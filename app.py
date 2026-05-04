import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import time
import pandas as pd

# ====== 设置 NLTK 数据目录 ======
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

try:
    import nltk
    from nltk import pos_tag, word_tokenize
    nltk.data.path.insert(0, nltk_data_dir)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ====== 强制使用 punkt ======
def ensure_nltk_punkt():
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            try:
                nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
            except:
                pass


# ====== 缓存加载 spaCy 模型 ======
@st.cache_resource
def load_spacy_model(model_name):
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        return None


class SyntaxAnalyzer:
    def __init__(self):
        self.offline_mode = False
    
    def get_model(self, language, force_offline=False):
        if force_offline:
            self.offline_mode = True
            return self.get_backup_parser(language)
        
        if not SPACY_AVAILABLE:
            self.offline_mode = True
            return self.get_backup_parser(language)
        
        model_name = 'en_core_web_sm' if language == 'en' else 'zh_core_web_sm'
        nlp = load_spacy_model(model_name)
        
        if nlp is not None:
            return nlp
        else:
            self.offline_mode = True
            return self.get_backup_parser(language)
    
    def get_backup_parser(self, language):
        jieba_available = False
        if language == 'zh' and JIEBA_AVAILABLE:
            try:
                jieba.initialize()
                jieba_available = True
            except:
                pass
        return {'type': 'backup', 'language': language, 'jieba_available': jieba_available}
    
    def run_dependency_analysis(self, text, nlp):
        result = {'tokens': [], 'svg': None}
        if nlp is None:
            return result
        
        if isinstance(nlp, dict) and nlp.get('type') == 'backup':
            if NLTK_AVAILABLE:
                try:
                    language = nlp.get('language', 'en')
                    jieba_available = nlp.get('jieba_available', False)
                    
                    if language == 'zh' and jieba_available:
                        tokens = jieba.lcut(text)
                    else:
                        tokens = word_tokenize(text)
                    
                    if language == 'zh':
                        tags = []
                        for token in tokens:
                            if len(token) == 1:
                                if token in '的了是在有和我他她它不人都一上们个中到大也很你说要去来会着没有看天过着出就好自己这':
                                    tags.append((token, 'PART'))
                                elif token in '我你他她它我们你们他们':
                                    tags.append((token, 'PRON'))
                                elif token in '是有在':
                                    tags.append((token, 'VERB'))
                                else:
                                    tags.append((token, 'NOUN'))
                            else:
                                tags.append((token, 'NOUN'))
                    else:
                        tags = pos_tag(tokens)
                    
                    for i, (token, pos) in enumerate(tags):
                        result['tokens'].append({
                            'text': token, 'pos': pos,
                            'head': i-1 if i > 0 else 0,
                            'deprel': 'nsubj' if i == 0 else 'dobj' if i == 1 else 'dep'
                        })
                    result['svg'] = self._generate_simple_dep_svg(result['tokens'])
                except Exception as e:
                    st.warning(f"离线模式分析失败: {str(e)}")
        elif SPACY_AVAILABLE:
            try:
                doc = nlp(text)
                result['svg'] = displacy.render(doc, style='dep', jupyter=False)
                for token in doc:
                    result['tokens'].append({
                        'text': token.text, 'pos': token.pos_,
                        'tag': token.tag_, 'head': token.head.i,
                        'deprel': token.dep_
                    })
            except Exception as e:
                st.warning(f"spaCy处理失败: {str(e)}")
        return result
    
    def _generate_simple_dep_svg(self, tokens):
        svg = '<svg width="600" height="200" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#161b22"/>'
        for i, token in enumerate(tokens):
            x = 50 + i * 100
            svg += f'<text x="{x}" y="150" fill="white" text-anchor="middle">{token["text"]}</text>'
            svg += f'<text x="{x}" y="170" fill="gray" font-size="12" text-anchor="middle">{token["pos"]}</text>'
        svg += '</svg>'
        return svg
    
    def extract_arguments(self, text, nlp):
        data = []
        if nlp and not isinstance(nlp, dict) and SPACY_AVAILABLE:
            try:
                doc = nlp(text)
                for token in doc:
                    if token.dep_ in ['nsubj', 'dobj', 'pobj', 'ROOT']:
                        data.append({
                            'Token': token.text, 'POS': token.pos_,
                            'Dependency': token.dep_, 'Head': token.head.text,
                            'Semantic Role': self._get_semantic_role(token.dep_)
                        })
            except Exception as e:
                st.warning(f"论元提取失败: {str(e)}")
        return pd.DataFrame(data) if data else pd.DataFrame(columns=['Token', 'POS', 'Dependency', 'Head', 'Semantic Role'])
    
    def _get_semantic_role(self, deprel):
        roles = {'nsubj': '施事/主语', 'dobj': '受事/直接宾语', 'pobj': '介词宾语', 'ROOT': '谓词核心'}
        return roles.get(deprel, '其他')


def main():
    st.set_page_config(page_title="句法透视仪 2.0", layout="wide")
    
    st.markdown("""
    <style>
        .stApp { background-color: #0d1117; color: #e6edf3; }
        .card { background-color: #161b22; border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1, h2, h3 { color: #00e676 !important; }
        .stButton>button { background-color: #00e676; color: #000; border-radius: 8px; font-weight: bold; transition: all 0.3s; }
        .stButton>button:hover { background-color: #00c853; transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 句法透视仪 2.0")
    
    with st.expander("📚 句法分析知识点", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["基础理论", "成分句法", "依存句法", "评估应用"])
        with tab1:
            st.write("**句法分析**：解析句子结构的关键技术")
            st.write("- 定义：NLP中解析句子结构的关键技术")
            st.write("- 核心作用：词序规则、短语结构规则、层级结构")
        with tab2:
            st.write("**成分句法分析**")
            st.write("- 核心思想：句子由嵌套短语结构组成")
            st.write("- 关键技术：短语结构规则(S→NP VP)、CYK算法、PCFG")
        with tab3:
            st.write("**依存句法分析**")
            st.write("- 核心思想：中心词与依存关系")
            st.write("- 关键概念：中心词(Head)、依存关系类型")
        with tab4:
            st.write("**评估指标**：PARSEVAL、UAS、LAS")
            st.write("**经典语料库**：PTB、CTB、UD")

    col1, col2 = st.columns(2)
    with col1:
        lang = st.radio("选择语言", ["English", "中文"], horizontal=True)
    with col2:
        mode = st.radio("运行模式", ["🚀 离线模式（推荐）", "📥 尝试加载spaCy模型"], horizontal=True)
    
    force_offline = (mode == "🚀 离线模式（推荐）")

    default_text = "The boy saw the man with the telescope" if lang == "English" else "咬死了猎人的狗"
    
    text = st.text_area("📝 输入文本", value=default_text, height=100)
    analyze_button = st.button("🔍 开始分析", type="primary")

    if analyze_button or text != default_text:
        if text.strip():
            with st.spinner("分析中..."):
                time.sleep(0.3)
                analyzer = SyntaxAnalyzer()
                lang_code = 'en' if lang == "English" else 'zh'
                nlp = analyzer.get_model(lang_code, force_offline=force_offline)
                dep_result = analyzer.run_dependency_analysis(text, nlp)
                args_df = analyzer.extract_arguments(text, nlp)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("依存句法分析")
            if analyzer.offline_mode:
                st.info("当前使用离线模式")
                if lang == "中文" and JIEBA_AVAILABLE:
                    st.success("✅ jieba分词器已启用")
                else:
                    st.warning("⚠️ jieba未安装，使用NLTK分词")
            else:
                st.success("✅ spaCy模型已加载")
            if dep_result['svg']:
                st.components.v1.html(dep_result['svg'], height=400, scrolling=True)
            if dep_result['tokens']:
                st.table(dep_result['tokens'])
            else:
                st.info("未能提取依存关系")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("核心论元提取")
            if not args_df.empty:
                st.dataframe(args_df, width='stretch')
                st.download_button("导出CSV", args_df.to_csv(index=False), "arguments.csv")
            else:
                st.info("未提取到核心论元或使用离线模式")
            st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🕵️ 歧义侦探实验室", expanded=False):
        case = st.selectbox("选择案例", [
            "The boy saw the man with the telescope",
            "Fruit flies like a banana",
            "南京市长江大桥",
            "咬死了猎人的狗"
        ])
        st.write(f"**歧义分析**: {case}")
        if case == "The boy saw the man with the telescope":
            st.write("- 解释1：男孩用望远镜看男人 (PP 修饰 VP)")
            st.write("- 解释2：男孩看拿着望远镜的男人 (PP 修饰 NP)")
            st.info("介词短语 'with the telescope' 可以修饰动词 'saw'（工具），也可以修饰名词 'man'（属性）")
        elif case == "Fruit flies like a banana":
            st.write("- 解释1：果蝇喜欢香蕉 (flies=名词)")
            st.write("- 解释2：水果像香蕉一样飞行 (flies=动词)")
            st.info("'Fruit flies' 既可以是名词短语（果蝇），也可以是 'fruit' + 'flies' 动词短语")
        elif case == "南京市长江大桥":
            st.write("- 解释1：南京市-长江大桥 (南京市的长江大桥)")
            st.write("- 解释2：南京市长-江大桥 (南京市长叫江大桥)")
            st.info(" '南京市' 和 '长江大桥' 之间存在组合歧义")
        elif case == "咬死了猎人的狗":
            st.write("- 解释1：（咬死了猎人）的狗 (狗咬死了猎人)")
            st.write("- 解释2：咬死了（猎人的狗）(猎人被咬死了狗)")
            st.info("动词 '咬死' 的主语/宾语归属存在歧义")


if __name__ == "__main__":
    ensure_nltk_punkt()
    main()
