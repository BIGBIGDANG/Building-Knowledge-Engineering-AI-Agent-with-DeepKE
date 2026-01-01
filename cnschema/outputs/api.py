import json
import requests
from volcengine.base.Request import Request

g_knowledge_base_domain = ""
apikey = ""  # 建议改成环境变量
SERVICE_ID = ""


def prepare_request(method, path, params=None, data=None, doseq=0):
    if params:
        for key in params:
            if isinstance(params[key], (int, float, bool)):
                params[key] = str(params[key])
            elif isinstance(params[key], list):
                if not doseq:
                    params[key] = ",".join(params[key])

    r = Request()
    r.set_shema("https")  
    r.set_method(method)
    r.set_connection_timeout(30)
    r.set_socket_timeout(60)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json;charset=UTF-8",
        "Host": g_knowledge_base_domain,
        "Authorization": f"Bearer {apikey}",
    }
    r.set_headers(headers)
    if params:
        r.set_query(params)
    r.set_host(g_knowledge_base_domain)
    r.set_path(path)
    if data is not None:
        r.set_body(json.dumps(data, ensure_ascii=False))
    return r


def call_knowledge_service_chat(messages, stream=False):
    method = "POST"
    path = "/api/knowledge/service/chat"

    request_params = {
        "service_resource_id": SERVICE_ID,
        "messages": messages,
        "stream": stream
    }

    info_req = prepare_request(method=method, path=path, data=request_params)

    # 用 https
    url = f"https://{g_knowledge_base_domain}{info_req.path}"

    rsp = requests.request(
        method=info_req.method,
        url=url,
        headers=info_req.headers,
        data=info_req.body.encode("utf-8"),
        timeout=60,
    )
    rsp.encoding = "utf-8"
    rsp.raise_for_status()
    return rsp.json()


def extract_answer(resp_json):
    data = resp_json.get("data", {})
    # 常见：问答型可能有 generated_answer
    if isinstance(data, dict) and data.get("generated_answer"):
        return data["generated_answer"]
    # 兜底：直接返回原始 json 字符串
    return json.dumps(resp_json, ensure_ascii=False, indent=2)


def main():
    
    history = []

    print("KnowledgeBase CLI Chat. 输入 exit/quit 退出。\n")

    while True:
        query = input("你：").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break

        # 组装 messages：把历史 + 本轮 user
        messages = history + [{"role": "user", "content": query}]

        try:
            resp = call_knowledge_service_chat(messages, stream=False)
        except Exception as e:
            print(f"\n[ERROR] 请求失败：{e}\n")
            continue

        answer = extract_answer(resp)
        print(f"\n助手：{answer}\n")

        # 写回历史：用于多轮
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

