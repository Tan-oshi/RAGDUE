"""
Gradio Web UI - Giao diện chat cho sinh viên.
Chỉ có cửa sổ chat, tối giản, thân thiện.
"""
import json
import logging
import random
import re
import threading
import time
from pathlib import Path

import gradio as gr

from .main import RAGPipeline, MissingTemporalReference

try:
    from .config import DEFAULT_TOP_K, DEFAULT_ALPHA, GRADIO_PORT, CHAT_HISTORY_PATH
except ImportError:
    from config import DEFAULT_TOP_K, DEFAULT_ALPHA, GRADIO_PORT, CHAT_HISTORY_PATH


def _detect_query_intent(query: str) -> str:
    """Simple query intent detection for greetings/thanks/farewell."""
    q = query.lower()
    if re.search(r"chào|hello|hi\s|hey|xin\s*chào|good\s*morning", q):
        return "greeting"
    if re.search(r"cảm\s*ơn|cảm\s*on|thank|thanks|thank\s*you", q):
        return "thanks"
    if re.search(r"hẹn\s*gặp\s*lại|tạm\s*biệt|bye|goodbye|good\s*night", q):
        return "farewell"
    return "info_seeking"

logger = logging.getLogger("rag_due_ui")

# --- Timeout settings ---
RAG_TIMEOUT_SECONDS = 12

# --- Pre-built answers for non-RAG queries ---
GREETING_ANSWERS = [
    "Xin chào! Mình là trợ lý lịch công tác của Trường Đại học Kinh tế, ĐH Đà Nẵng. Bạn cần hỏi gì hôm nay?",
    "Chào bạn! Rất vui được hỗ trợ bạn tra cứu lịch công tác. Bạn cứ hỏi nhé!",
]
THANKS_ANSWERS = [
    "Không có gì! Còn câu hỏi nào khác không?",
    "Cảm ơn bạn! Nếu cần hỏi thêm, cứ nhắn nhé!",
]
FAREWELL_ANSWERS = [
    "Tạm biệt! Chúc bạn một ngày tốt lành!",
    "Hẹn gặp lại bạn!",
]

# --- Global state ---
_pipeline: RAGPipeline | None = None
_chat_sessions: dict[str, list[dict[str, str]]] = {}
_current_session = "default"
_force_recreate = False


def set_recreate_on_startup(value: bool) -> None:
    global _force_recreate
    _force_recreate = value


def get_pipeline() -> RAGPipeline:
    global _pipeline, _force_recreate
    if _pipeline is None:
        logger.info("Khởi tạo RAG Pipeline...")
        _pipeline = RAGPipeline()
        if _force_recreate:
            import pathlib
            _project_root = pathlib.Path(__file__).resolve().parent.parent
            jsonl_path = str(_project_root / "data/raw/master_lich_tuan.jsonl")
            logger.info(f"Force recreate: re-ingesting from {jsonl_path}")
            _pipeline.ingest(jsonl_path=jsonl_path, recreate_collection=True)
            _force_recreate = False
        else:
            _pipeline._ensure_loaded()
        logger.info(f"Pipeline ready: {len(_pipeline._available_weeks)} tuần loaded")
    return _pipeline


def chat(message: str, history: list[dict]):
    """
    Xử lý một lượt chat. Generator để streaming tokens từng phần.
    """
    global _current_session, _chat_sessions
    _current_session = "default"
    _chat_sessions.setdefault("default", [])

    query_intent = _detect_query_intent(message)
    logger.info(f"Query intent: {query_intent}")

    # --- Non-RAG intents ---
    if query_intent in ("greeting", "thanks", "farewell"):
        answers_map = {
            "greeting": GREETING_ANSWERS,
            "thanks": THANKS_ANSWERS,
            "farewell": FAREWELL_ANSWERS,
        }
        answer = random.choice(answers_map[query_intent])
        _chat_sessions["default"].append({"role": "user", "content": message})
        _chat_sessions["default"].append({"role": "assistant", "content": answer})
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        yield history, ""
        return

    # --- RAG pipeline ---
    pipeline = get_pipeline()
    result_holder: dict = {}

    def _run_retrieval():
        try:
            answer, results, scope, temporal_scope_val = pipeline.ask(
                question=message,
                session_history=_chat_sessions["default"],
                top_k=DEFAULT_TOP_K,
                alpha=DEFAULT_ALPHA,
                week_filter=None,
            )
            result_holder["answer"] = answer
            result_holder["results"] = results
            result_holder["scope"] = scope
            result_holder["temporal_scope"] = temporal_scope_val
            result_holder["error"] = None
            result_holder["temporal_missing"] = False
        except MissingTemporalReference as e:
            result_holder["answer"] = None
            result_holder["results"] = []
            result_holder["scope"] = None
            result_holder["error"] = None
            result_holder["temporal_missing"] = True
            result_holder["temporal_message"] = str(e)
        except Exception as e:
            logger.error(f"Pipeline exception: {e}", exc_info=True)
            result_holder["answer"] = None
            result_holder["results"] = []
            result_holder["scope"] = None
            result_holder["error"] = str(e)

    thread = threading.Thread(target=_run_retrieval)
    thread.start()

    # Yield immediately with question + loading placeholder
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Đang tìm kiếm..."})
    yield history, ""

    thread.join()

    while thread.is_alive():
        time.sleep(0.2)

    # --- Process results ---
    answer = result_holder.get("answer")
    error_msg = result_holder.get("error")
    temporal_missing = result_holder.get("temporal_missing", False)

    if temporal_missing:
        history[-1] = {
            "role": "assistant",
            "content": result_holder.get(
                "temporal_message",
                "Vui lòng hỏi kèm thông tin thời gian (ví dụ: 'Tuần này có cuộc họp gì?', 'Thứ 2 tuần sau').",
            ),
        }
        yield history, ""
        return

    if error_msg:
        history[-1] = {"role": "assistant", "content": f"Đã xảy ra lỗi: {error_msg}"}
        yield history, ""
        return

    if not answer:
        scope = result_holder.get("scope", "")
        scope_hint = f" (phạm vi: {scope})" if scope else ""
        history[-1] = {
            "role": "assistant",
            "content": f"Không tìm thấy thông tin{scope_hint}. "
                       f"Bạn có thể thử hỏi cụ thể hơn (ví dụ: 'Thứ 2 tuần này có lịch học gì?').",
        }
        yield history, ""
        return

    # --- Stream the answer token-by-token ---
    from .generation import build_rag_prompt, build_rag_prompt_with_history, format_context_from_results, format_chat_history, SYSTEM_PROMPT
    from google.genai import types

    results = result_holder["results"]
    scope = result_holder["scope"]
    temporal_scope = result_holder.get("temporal_scope")

    # Build prompt (same as generate_answer_with_history but without calling Gemini yet)
    context = format_context_from_results(results)
    if _chat_sessions["default"]:
        history_str = format_chat_history(_chat_sessions["default"])
        prompt = build_rag_prompt_with_history(
            question=message,
            context=context,
            chat_history=history_str,
            scope_context=scope,
            temporal_scope=temporal_scope,
        )
    else:
        prompt = build_rag_prompt(question=message, context=context, scope_context=scope, temporal_scope=temporal_scope)

    # Get Gemini client and stream
    from .generation import get_gemini_client
    client = get_gemini_client()

    try:
        from ..config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
    except ImportError:
        from config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

    partial = ""
    response_stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_TOKENS,
        ),
    )

    for chunk in response_stream:
        if chunk.text:
            partial += chunk.text
            history[-1] = {"role": "assistant", "content": partial}
            yield history, ""

    # Final: save to session history
    _chat_sessions["default"].append({"role": "user", "content": message})
    _chat_sessions["default"].append({"role": "assistant", "content": partial})
    yield history, ""


def clear_history():
    """Xoá toàn bộ lịch sử chat."""
    global _chat_sessions
    _chat_sessions["default"] = []
    return []


def launch_gradio(share: bool = False, port: int = GRADIO_PORT):
    """Khởi động Gradio app — giao diện chat thuần túy."""

    # Loading popup JS
    popup_html = """
    <div id="rag-popup" style="
        display:none;
        position:fixed;top:0;left:0;width:100%;height:100%;
        background:rgba(0,0,0,0.5);
        z-index:9999;justify-content:center;align-items:center;
    ">
        <div style="
            background:white;border-radius:16px;padding:32px 40px;
            max-width:380px;text-align:center;
            box-shadow:0 8px 32px rgba(0,0,0,0.25);
            font-family:system-ui,-apple-system,sans-serif;
        ">
            <div style="
                width:44px;height:44px;border:4px solid rgba(66,133,244,0.2);
                border-top-color:#4285f4;border-radius:50%;
                animation:spin 0.7s linear infinite;margin:0 auto 16px;
            "></div>
            <h3 style="margin:0 0 8px;color:#1a1a1a;font-size:18px;">Đang tìm kiếm...</h3>
            <p style="margin:0;color:#666;font-size:13px;line-height:1.5;">
                Vui lòng đợi trong giây lát
            </p>
        </div>
    </div>
    <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
    <script>
    (function() {
        var timer = null;
        function show(){var p=document.getElementById('rag-popup');if(p)p.style.display='flex';}
        function hide(){var p=document.getElementById('rag-popup');if(p)p.style.display='none';}
        function reset(){hide();if(timer)clearTimeout(timer);timer=setTimeout(show, 8000);}
        window.addEventListener('DOMContentLoaded',function(){
            var iv=setInterval(function(){
                document.querySelectorAll('form').forEach(function(f){
                    if(!f._r){f._r=true;f.addEventListener('submit',reset);}
                });
                document.querySelectorAll('button').forEach(function(b){
                    if(b.textContent.includes('Xoá')||b.textContent.includes('Clear')){
                        b.addEventListener('click',hide);
                    }
                });
            },500);
            setTimeout(reset, 8000);
        });
    })();
    </script>
    """

    with gr.Blocks(title="Trợ lý Lịch Công Tác") as demo:
        gr.HTML(popup_html)

        # Header
        gr.HTML("""
        <div style="text-align:center;padding:16px 0 8px;font-family:system-ui,sans-serif;">
            <h1 style="margin:0 0 4px;font-size:22px;color:#1a1a2e;">
                📅 Trợ lý Lịch Công Tác
            </h1>
            <p style="margin:0;color:#666;font-size:13px;">
                Trường Đại học Kinh tế, Đại học Đà Nẵng
            </p>
        </div>
        """)

        # Chat area
        chatbot = gr.Chatbot(
            height=520,
        )

        # Input row
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Hỏi về lịch công tác (ví dụ: Tuần này có cuộc họp gì?)",
                scale=8,
                submit_btn="Gửi",
            )
            clear_btn = gr.Button("🗑️", scale=1, size="sm")

        # Footer hint
        gr.HTML("""
        <div style="text-align:center;padding:8px 0 4px;color:#999;font-size:12px;font-family:system-ui,sans-serif;">
            💡 Gợi ý: "Tuần này có những cuộc họp nào?" · "Thứ 2 tuần sau có lịch gì?" · "Lịch tháng 5"
        </div>
        """)

        # Event handlers
        msg_box.submit(
            chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box],
        )
        clear_btn.click(fn=clear_history, outputs=[chatbot])

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        #chat-4, #chat-5 { display: none !important; }
        .message.bot { background: #e8f0fe !important; }
        """,
    )
