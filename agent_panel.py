import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import Optional
from movie_tool import get_tools, query_movie_db
from chart_tool import get_chart_tool, validate_chart


# ── State ──

DEFAULT_STATE = {
    "agent_phase": "idle",
    "agent_events": [],
    "agent_messages": [],
    "agent_tools": [],
    "agent_df": None,
    "agent_chart_specs": [],
    "agent_pending_message": None,
    "agent_rewrite_event_index": None,
}

def get_state(key):
    return st.session_state.get(key, DEFAULT_STATE[key])

def set_state(key, value):
    st.session_state[key] = value

def restart_agent(user_question, filtered_df, show_chart=False):
    set_state("agent_phase", "thinking")
    set_state("agent_events", [])
    set_state("agent_chart_specs", [])
    set_state("agent_pending_message", None)

    tools = get_tools(filtered_df)
    system_content = "You are a data analyst with access to a tool that executes Python code on a movie database."

    if show_chart:
        tools.append(get_chart_tool())
        system_content += " After computing the data, create a chart using a Vega-Lite specification."

    set_state("agent_messages", [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question},
    ])
    set_state("agent_tools", tools)
    set_state("agent_df", filtered_df)


# ── Logic ──

def run_step(client):
    phase = get_state("agent_phase")
    messages = get_state("agent_messages")

    if phase == "thinking":
        class Reasoning(BaseModel):
            reason: str = Field(description="Your reasoning about what you know so far and what to do next")
            use_tool: bool = Field(description="True if you need to run code or create a chart, False if you can give the final answer")
            answer: Optional[str] = Field(default=None, description="Your final answer in one short paragraph. Only provide when use_tool is False.")

        response = client.chat.completions.parse(
            model="gpt-4o-mini", messages=messages, response_format=Reasoning,
        )
        reasoning = response.choices[0].message.parsed
        messages.append({"role": "assistant", "content": reasoning.reason})

        if reasoning.use_tool:
            get_state("agent_events").append({"type": "thought", "thought": reasoning.reason})
            set_state("agent_phase", "acting")
        else:
            get_state("agent_events").append({"type": "answer", "thought": reasoning.reason, "answer": reasoning.answer})
            set_state("agent_phase", "done")

    elif phase == "acting":
        tools = get_state("agent_tools")

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools, parallel_tool_calls=False,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            set_state("agent_phase", "done")
            return

        set_state("agent_pending_message", msg)
        set_state("agent_phase", "awaiting_approval")

def execute_pending_tools():
    messages = get_state("agent_messages")
    df = get_state("agent_df")
    pending_msg = get_state("agent_pending_message")

    rewind_point = len(messages)
    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        args = json.loads(tc.function.arguments)

        if tc.function.name == "QueryMovieDB":
            result = query_movie_db(args["code"], df)
            get_state("agent_events").append({
                "type": "action", "name": tc.function.name,
                "code": args["code"], "result": result,
                "_rewind_point": rewind_point,
            })
        elif tc.function.name == "CreateChart":
            spec, result = validate_chart(args["vega_lite_spec"])
            if spec:
                get_state("agent_chart_specs").append(spec)
            get_state("agent_events").append({
                "type": "chart", "name": tc.function.name,
                "spec_str": args["vega_lite_spec"], "result": result,
                "_rewind_point": rewind_point,
            })

        messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")

def reject_pending_tools(feedback):
    messages = get_state("agent_messages")
    pending_msg = get_state("agent_pending_message")

    rejection_msg = "User rejected this action."
    if feedback:
        rejection_msg += f" User feedback: {feedback}"
    else:
        rejection_msg += " Try a different approach."

    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        get_state("agent_events").append({
            "type": "rejected", "name": tc.function.name,
            "feedback": feedback,
        })
        messages.append({"role": "tool", "content": rejection_msg, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")


# ── Rendering ──

TOOL_DESCRIPTIONS = {
    "QueryMovieDB": "Runs Python code on the movie database to compute a result.",
    "CreateChart": "Creates a Vega-Lite chart visualization from the data.",
}

def _get_latest_thought():
    events = get_state("agent_events")
    for event in reversed(events):
        if event["type"] == "thought":
            return event["thought"]
    return None

def _render_inline_edit():
    st.info("Describe how you want the code/spec changed. The agent will regenerate it.")
    edit_prompt = st.text_input("What changes do you want?", key="edit_prompt")
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.button("Submit Edit", type="primary", use_container_width=True)
    with col2:
        cancelled = st.button("Cancel", use_container_width=True)
    return {"edit_submitted": submitted, "edit_prompt": edit_prompt, "edit_cancelled": cancelled}

def render_events(allow_rewind=False, inline_edit_index=None):
    clicked_index = None
    inline_edit_actions = None
    for i, event in enumerate(get_state("agent_events")):
        if event["type"] == "thought":
            st.markdown(f"**Thought:** {event['thought']}")
        elif event["type"] == "action":
            st.markdown(f"**Action:** `{event['name']}`")
            st.code(event["code"], language="python")
            st.markdown("**Observation:**")
            st.code(event["result"], language="text")
            if allow_rewind:
                if st.button("Edit", key=f"rewind_{i}"):
                    clicked_index = i
            if inline_edit_index is not None and i == inline_edit_index:
                inline_edit_actions = _render_inline_edit()
            st.divider()
        elif event["type"] == "chart":
            st.markdown(f"**Action:** `{event['name']}`")
            st.code(event["spec_str"], language="json")
            st.markdown("**Observation:**")
            st.code(event["result"], language="text")
            if allow_rewind:
                if st.button("Edit", key=f"rewind_{i}"):
                    clicked_index = i
            if inline_edit_index is not None and i == inline_edit_index:
                inline_edit_actions = _render_inline_edit()
            st.divider()
        elif event["type"] == "rejected":
            st.markdown(f"**Rejected:** `{event['name']}`")
            if event.get("feedback"):
                st.text(f"Feedback: {event['feedback']}")
            st.divider()
        elif event["type"] == "edited":
            st.markdown(f"**Edit:** {event['prompt']}")
            st.divider()
        elif event["type"] == "answer":
            st.markdown(f"**Thought:** {event['thought']}")
    return clicked_index, inline_edit_actions

def render_pending_approval():
    st.warning("The agent wants to perform the following action:")
    thought = _get_latest_thought()
    if thought:
        st.markdown(f"**Agent's intent:** {thought}")
    for tc in get_state("agent_pending_message").tool_calls:
        args = json.loads(tc.function.arguments)
        desc = TOOL_DESCRIPTIONS.get(tc.function.name, "")
        st.markdown(f"**Tool:** `{tc.function.name}` — {desc}")
        if tc.function.name == "QueryMovieDB":
            st.code(args["code"], language="python")
        elif tc.function.name == "CreateChart":
            st.code(args["vega_lite_spec"], language="json")

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        approved = st.button("Approve", type="primary", use_container_width=True)
    with btn_col2:
        edited = st.button("Edit", use_container_width=True)
    with btn_col3:
        rejected = st.button("Reject", use_container_width=True)
    return approved, edited, rejected

def render_pending_feedback():
    feedback = st.text_input(
        "Why are you rejecting? Tell the agent what to do instead:",
        key="reject_feedback",
    )
    submitted = st.button("Submit Rejection", use_container_width=True)
    return submitted, feedback

def render_pending_edit():
    st.info("Describe how you want the code/spec changed. The agent will regenerate it.")
    thought = _get_latest_thought()
    if thought:
        st.markdown(f"**Agent's intent:** {thought}")
    pending_msg = get_state("agent_pending_message")
    for tc in pending_msg.tool_calls:
        args = json.loads(tc.function.arguments)
        desc = TOOL_DESCRIPTIONS.get(tc.function.name, "")
        st.markdown(f"**Tool:** `{tc.function.name}` — {desc}")
        if tc.function.name == "QueryMovieDB":
            st.code(args["code"], language="python")
        elif tc.function.name == "CreateChart":
            st.code(args["vega_lite_spec"], language="json")

    edit_prompt = st.text_input(
        "What changes do you want?",
        key="edit_prompt",
    )
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.button("Submit Edit", type="primary", use_container_width=True)
    with col2:
        cancelled = st.button("Cancel", use_container_width=True)
    return submitted, edit_prompt, cancelled

def edit_pending_tools(edit_prompt):
    messages = get_state("agent_messages")
    pending_msg = get_state("agent_pending_message")

    edit_msg = f"User wants changes to the proposed code: {edit_prompt}. Regenerate the tool call with these changes."

    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        messages.append({"role": "tool", "content": edit_msg, "tool_call_id": tc.id})

    get_state("agent_events").append({"type": "edited", "prompt": edit_prompt})
    set_state("agent_pending_message", None)
    set_state("agent_phase", "acting")

def rewind_and_edit(event_index, edit_prompt):
    events = get_state("agent_events")
    rewind_point = events[event_index]["_rewind_point"]

    messages = get_state("agent_messages")
    del messages[rewind_point:]
    del events[event_index:]

    chart_specs = [
        validate_chart(e["spec_str"])[0]
        for e in events if e["type"] == "chart"
    ]
    set_state("agent_chart_specs", [s for s in chart_specs if s])

    events.append({"type": "edited", "prompt": edit_prompt})
    messages.append({"role": "user", "content": f"User wants changes: {edit_prompt}. Regenerate the tool call."})
    set_state("agent_rewrite_event_index", None)
    set_state("agent_phase", "acting")

def render_panel():
    st.subheader("Analysis Results")
    container = st.container(height=600)
    actions = {}
    with container:
        phase = get_state("agent_phase")

        if phase == "idle":
            st.info("Enter a question and click 'Analyze' to see results.")

        elif phase in ("thinking", "acting"):
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            st.spinner("Agent is thinking...")

        elif phase == "awaiting_approval":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            approved, edited, rejected = render_pending_approval()
            actions = {"approved": approved, "edited": edited, "rejected": rejected}

        elif phase == "awaiting_edit":
            rewrite_index = get_state("agent_rewrite_event_index")
            if rewrite_index is not None:
                with st.expander("Agent Reasoning Trace", expanded=True):
                    _, inline_actions = render_events(inline_edit_index=rewrite_index)
                if inline_actions:
                    actions.update(inline_actions)
            else:
                with st.expander("Agent Reasoning Trace", expanded=True):
                    render_events()
                submitted, edit_prompt, cancelled = render_pending_edit()
                actions = {"edit_submitted": submitted, "edit_prompt": edit_prompt, "edit_cancelled": cancelled}

        elif phase == "awaiting_feedback":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            submitted, feedback = render_pending_feedback()
            actions = {"submitted": submitted, "feedback": feedback}

        elif phase == "done":
            with st.expander("Agent Reasoning Trace", expanded=False):
                clicked_index, _ = render_events(allow_rewind=True)
            if clicked_index is not None:
                actions["rewind_index"] = clicked_index
            events = get_state("agent_events")
            if events and events[-1].get("answer"):
                st.write("**Answer:**")
                st.write(events[-1]["answer"])
            for spec in get_state("agent_chart_specs"):
                st.vega_lite_chart(spec, use_container_width=True)

    return actions


# ── Lifecycle ──

def agent_panel(client, analyze_button, user_question, filtered_df, show_chart=False):
    # Phases: idle -> thinking <-> acting -> awaiting_approval -> thinking ... -> done
    #                                     -> awaiting_edit -> acting (regenerate) -> awaiting_approval ...
    #                                     -> awaiting_feedback -> thinking ... -> done
    #         done -> awaiting_edit (rewind) -> acting (regenerate) -> awaiting_approval ... -> done
    if analyze_button and user_question:
        restart_agent(user_question, filtered_df, show_chart)

    actions = render_panel()

    phase = get_state("agent_phase")
    if phase in ("thinking", "acting"):
        run_step(client)
        st.rerun()
    elif phase == "awaiting_approval":
        if actions.get("approved"):
            execute_pending_tools()
            st.rerun()
        elif actions.get("edited"):
            set_state("agent_phase", "awaiting_edit")
            st.rerun()
        elif actions.get("rejected"):
            set_state("agent_phase", "awaiting_feedback")
            st.rerun()
    elif phase == "awaiting_edit":
        if actions.get("edit_cancelled"):
            if get_state("agent_pending_message"):
                set_state("agent_phase", "awaiting_approval")
            else:
                set_state("agent_rewrite_event_index", None)
                set_state("agent_phase", "done")
            st.rerun()
        elif actions.get("edit_submitted") and actions.get("edit_prompt"):
            if get_state("agent_pending_message"):
                edit_pending_tools(actions["edit_prompt"])
            elif get_state("agent_rewrite_event_index") is not None:
                rewind_and_edit(get_state("agent_rewrite_event_index"), actions["edit_prompt"])
            st.rerun()
    elif phase == "done":
        if actions.get("rewind_index") is not None:
            set_state("agent_rewrite_event_index", actions["rewind_index"])
            set_state("agent_phase", "awaiting_edit")
            st.rerun()
    elif phase == "awaiting_feedback" and actions.get("submitted"):
        reject_pending_tools(actions.get("feedback", ""))
        st.rerun()
