import torch
import json
import copy
import numpy as np
from llm.utils.general.registry_factory import PARSER_REGISTRY, AUGMENTATION_REGISTRY


class NLPCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


@PARSER_REGISTRY.register('pretrain_bin')
class PretrainBinParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta
        self.pad_token_id = len(tokenizer) - 1

    def _pad_tokens(self, tokens):
        pad_tokens = [self.pad_token_id] * self.max_seq_length
        pad_tokens[:len(tokens)] = tokens
        return pad_tokens

    def get_cu_seqlens(self, tokens):
        # split by eos
        cu_seqlens = (np.where(tokens == self.tokenizer.eos_token_id)[0] + 1).tolist()
        if len(cu_seqlens) == 0:
            cu_seqlens.append(len(tokens))
        else:
            if cu_seqlens[-1] != len(tokens):
                cu_seqlens.append(len(tokens))
        cu_seqlens.insert(0, 0)
        return cu_seqlens

    def get_position_ids(self, cu_seqlens):
        position_ids = []
        for i in range(1, len(cu_seqlens)):
            position_ids.extend(list(range(cu_seqlens[i] - cu_seqlens[i - 1])))
        return position_ids

    def __call__(self, meta):
        path = meta['path']
        bin_index = meta['bin_index']
        start, end = bin_index[1], bin_index[2]
        tokens = np.load(path)[start:end]
        if len(tokens) < self.max_seq_length:
            tokens = self._pad_tokens(tokens)
        cu_seqlens = self.get_cu_seqlens(tokens)
        position_ids = self.get_position_ids(cu_seqlens)
        labels = np.copy(tokens)
        input_ids = torch.LongTensor(tokens)
        labels = torch.LongTensor(labels)
        cu_seqlens = torch.LongTensor(cu_seqlens)
        position_ids = torch.LongTensor(position_ids)
        results = {'input_ids': input_ids, 'labels': labels, "cu_seqlens": cu_seqlens, "position_ids": position_ids}
        return results


@PARSER_REGISTRY.register('tools')
class ToolParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 drop_meta=False,
                 prompt_template={},
                 use_system=True,
                 use_knowledge=True,
                 ensure_ascii=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta
        self.system_prompt = prompt_template.get('sytem_prompt', "<system>: ")
        self.user_prompt = prompt_template.get('user_prompt', "<user>: ")
        self.assistant_prompt = prompt_template.get("assistant_prompt", "<assistant>: ")
        self.tool_calls_start = prompt_template.get('tool_calls_start', "<tool_calls_start>\n")
        self.tool_calls_end = prompt_template.get('tool_calls_end', "<tool_calls_end>\n")
        self.tool_response_prompt = prompt_template.get('tool_response_prompt', "<tool_response>:\n")
        self.tool_define = prompt_template.get("tool_define", "<tools_define>:\n")
        self.knowledge_prompt = prompt_template.get('knowledge_prompt', "<knowledge>: ")
        self.use_system = use_system
        self.use_knowledge = use_knowledge
        self.ensure_ascii = ensure_ascii

    def __call__(self, meta):
        if 'input' in meta:
            messages = meta['input'].get('messages', [])
            tools = meta['input'].get('tools', [])
        elif "messages" in meta:
            messages = meta['messages']
            tools = meta['tools']
        else:
            messages = meta
            tools = []
        tokens = []
        labels = []
        system = ''
        if len(tools) > 0:
            system += self.tool_define
        for tool in tools:
            system += (json.dumps(tool['function'], ensure_ascii=self.ensure_ascii) + "\n")
        tokenized_tool = self.tokenizer(system, return_attention_mask=False)['input_ids']
        tokens.extend(tokenized_tool)
        labels.extend([self.ignore_index] * len(tokenized_tool))

        for item in messages:
            if self.use_system:
                if item['role'] == 'system' and item['content'] != '':
                    system = f"{self.system_prompt}{item['content']}\n"
                    tokenized_system = self.tokenizer(system, return_attention_mask=False,
                                                      add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_system)
                    labels.extend([self.ignore_index] * len(tokenized_system))
            if item['role'] == 'user':
                user_info = f"{self.user_prompt}{item['content']}\n"
                tokenized_user = self.tokenizer(user_info, return_attention_mask=False,
                                                add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_user)
                labels.extend([self.ignore_index] * len(tokenized_user))
            if self.use_knowledge:
                if item['role'] == 'knowledge':
                    knowledge_info = f"{self.knowledge_prompt}{item['content']}\n"
                    tokenized_knowledge = self.tokenizer(knowledge_info, return_attention_mask=False,
                                                         add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_knowledge)
                    labels.extend([self.ignore_index] * len(tokenized_knowledge))

            if item['role'] == 'assistant':
                assis_info = ''
                tokens_assistant_prompt = self.tokenizer(self.assistant_prompt, return_attention_mask=False,
                                                         add_special_tokens=False)['input_ids']
                tokens.extend(tokens_assistant_prompt)
                labels.extend([self.ignore_index] * len(tokens_assistant_prompt))

                if item['content']:
                    assis_info += f"{item['content']}"
                if 'tool_calls' in item and len(item['tool_calls']) > 0:
                    assis_info += self.tool_calls_start
                    for tool_call in item['tool_calls']:
                        assis_info += json.dumps(tool_call['function'], ensure_ascii=self.ensure_ascii) + "\n"
                    assis_info += self.tool_calls_end
                tokenized_assistant = self.tokenizer(assis_info, return_attention_mask=False,
                                                     add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_assistant)
                labels.extend(copy.deepcopy(tokenized_assistant))
                if not self.inference_mode:
                    tokens += [self.tokenizer.eos_token_id]
                    labels += [self.tokenizer.eos_token_id]

            if item['role'] == 'tool':
                response_info = f"{self.tool_response_prompt}{item['content']}"
                tokenized_tool = self.tokenizer(response_info, return_attention_mask=False,
                                                add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_tool)
                labels.extend([self.ignore_index] * len(tokenized_tool))
        if self.inference_mode:
            infer_tokens_assistant_prompt = self.tokenizer(self.assistant_prompt, return_attention_mask=False,
                                                           add_special_tokens=False)['input_ids']
            tokens.extend(infer_tokens_assistant_prompt)
            labels.extend([self.ignore_index] * len(infer_tokens_assistant_prompt))
            return tokens, []
        if self.keep_all_keys:
            labels = copy.deepcopy(tokens)
        else:
            if self.drop_meta and len(tokens) > self.max_seq_length:
                return None
            # drop question to avoid no loss
            tokens = tokens[-self.max_seq_length:]
            labels = labels[-self.max_seq_length:]
            input_ids = torch.LongTensor(tokens)
            labels = torch.LongTensor(labels)
            results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('intern_tools')
class InternToolParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 drop_meta=False,
                 prompt_template={},
                 use_system=True,
                 use_knowledge=True,
                 use_interpreter=True,
                 ensure_ascii=False,
                 tool_mode='merge'):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta
        self.conversation_start = prompt_template.get("conversation_start", "<|im_start|>")
        self.conversation_end = prompt_template.get("conversation_end", "<|im_end|>")
        self.action_start = prompt_template.get("action_start", "<|action_start|>")
        self.action_end = prompt_template.get("action_end", "<|action_end|>")
        # plugin tools
        self.plugin_prompt = prompt_template.get("plugin", "<|plugin|>")
        # interpreter
        self.interpreter_prompt = prompt_template.get("interpreter", "<|interpreter|>")

        self.use_system = use_system
        self.use_knowledge = use_knowledge
        self.use_interpreter = use_interpreter
        self.ensure_ascii = ensure_ascii
        self.tool_mode = tool_mode

    def __call__(self, meta):
        if 'input' in meta:
            messages = meta['input'].get('messages', [])
            tools = meta['input'].get('tools', [])
            interpreter = meta['input'].get('interpreter', [])
        elif "messages" in meta:
            messages = meta['messages']
            tools = meta.get('tools', [])
            interpreter = meta.get('interpreter', [])
        else:
            messages = meta
            tools = []
            interpreter = []
        tokens = []
        labels = []

        tokens.extend([self.tokenizer.bos_token_id])
        labels.extend([self.ignore_index])

        conversation_messages = messages
        if self.use_system:
            if messages[0]['role'] == 'system' and messages[0]['content'] != '':
                system = f"{self.conversation_start}system\n{messages[0]['content']}{self.conversation_end}\n"
                tokenized_system = self.tokenizer(system, return_attention_mask=False,
                                                  add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_system)
                labels.extend([self.ignore_index] * len(tokenized_system))
                # remove the system item
                conversation_messages = messages[1:]

        if self.use_interpreter and (len(interpreter) > 0):
            assert (len(interpreter) == 1) and (interpreter[0]["name"] == "python_interpreter"), "Only support python interpreter now!"     # noqa
            interpreter_str = f"{self.conversation_start}system name={self.interpreter_prompt}\n{interpreter[0]['description']}\n{self.conversation_end}\n"
            tokenized_interpreter = self.tokenizer(interpreter_str, return_attention_mask=False, add_special_tokens=False)['input_ids']
            tokens.extend(tokenized_interpreter)
            labels.extend([self.ignore_index] * len(tokenized_interpreter))
        if len(tools) > 0:
            plugin_tools = []
            for tool in tools:
                if not (self.use_interpreter and (tool['function']['name'] == "python_interpreter")):
                    plugin_tools.append(copy.deepcopy(tool['function']))
            if len(plugin_tools) > 0:
                if self.tool_mode == 'merge':
                    plugin_tools_str = json.dumps(plugin_tools, ensure_ascii=self.ensure_ascii)
                else:
                    plugin_tools_str = ''
                    for p_idx, tool in enumerate(plugin_tools):
                        plugin_tools_str += json.dumps(tool, ensure_ascii=self.ensure_ascii)
                        if p_idx != len(plugin_tools) - 1:
                            plugin_tools_str += '\n'
                plugin_tools_str = f"{self.conversation_start}system name={self.plugin_prompt}\n{plugin_tools_str}\n{self.conversation_end}\n"
                tokenized_plugin_tools = self.tokenizer(plugin_tools_str, return_attention_mask=False, add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_plugin_tools)
                labels.extend([self.ignore_index] * len(tokenized_plugin_tools))

        for item in conversation_messages:
            # assert item['role'] != 'system', "only allow system at the start of conversation"
            if self.use_knowledge:
                # do not support knowledge yet
                raise NotImplementedError
            if item['role'] == 'user':
                user_info = f"{self.conversation_start}user\n{item['content']}{self.conversation_end}\n"
                tokenized_user = self.tokenizer(user_info, return_attention_mask=False,
                                                add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_user)
                labels.extend([self.ignore_index] * len(tokenized_user))
            if item['role'] == 'assistant':
                assis_start = f"{self.conversation_start}assistant\n"
                tokens_assistant_start = self.tokenizer(assis_start, return_attention_mask=False,
                                                        add_special_tokens=False)['input_ids']
                tokens.extend(tokens_assistant_start)
                labels.extend([self.ignore_index] * len(tokens_assistant_start))

                assis_info = ""
                if item['content']:
                    assis_info = item['content']

                if 'tool_calls' in item and len(item['tool_calls']) > 0:
                    assis_info += self.action_start
                    for tool_call in item['tool_calls']:
                        if self.use_interpreter and 'name'in tool_call['function'] and tool_call['function']['name'] == "python_interpreter":
                            assis_info += f"{self.interpreter_prompt}\n{tool_call['function']['arguments']['code']}\n"      # noqa
                        else:
                            assis_info += f"{self.plugin_prompt}\n{json.dumps(tool_call['function'], ensure_ascii=self.ensure_ascii)}\n"
                    assis_info += self.action_end

                if not self.inference_mode:
                    assis_info += f"{self.conversation_end}\n"
                tokenized_assistant = self.tokenizer(assis_info, return_attention_mask=False,
                                                     add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_assistant)
                labels.extend(copy.deepcopy(tokenized_assistant))

            if item['role'] == 'tool':
                if self.use_interpreter and 'name' in item and item['name'] == "python_interpreter":
                    response_info = f"{self.conversation_start}environment name={self.interpreter_prompt}\n{item['content']}{self.conversation_end}\n"
                else:
                    response_info = f"{self.conversation_start}environment name={self.plugin_prompt}\n{item['content']}{self.conversation_end}\n"
                tokenized_response = self.tokenizer(response_info, return_attention_mask=False, add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_response)
                labels.extend([self.ignore_index] * len(tokenized_response))
        if self.inference_mode:
            infer_tokens_assistant_prompt = self.tokenizer(f"{self.conversation_start}assistant\n", return_attention_mask=False,
                                                           add_special_tokens=False)['input_ids']
            tokens.extend(infer_tokens_assistant_prompt)
            labels.extend([self.ignore_index] * len(infer_tokens_assistant_prompt))
            return tokens, []
        if self.keep_all_keys:
            labels = copy.deepcopy(tokens)
        else:
            if self.drop_meta and len(tokens) > self.max_seq_length:
                return None
            # drop question to avoid no loss
            tokens = tokens[-self.max_seq_length:]
            labels = labels[-self.max_seq_length:]
            input_ids = torch.LongTensor(tokens)
            labels = torch.LongTensor(labels)
            results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('preprocess')
class PreProcessParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta

    def __call__(self, meta):
        question = meta['inputs']
        answer = meta.get('outputs', "")
        tokenized_question = self.tokenizer(question, return_attention_mask=False)['input_ids']
        tokenized_answer = self.tokenizer(answer, return_attention_mask=False, add_special_tokens=False)['input_ids']
        if self.keep_all_keys:
            labels = tokenized_question + tokenized_answer
        else:
            labels = [self.ignore_index] * len(tokenized_question) + tokenized_answer
        if self.inference_mode:
            return tokenized_question + tokenized_answer, []
        else:
            tokenized_text = tokenized_question + tokenized_answer + [self.tokenizer.eos_token_id]
            labels = labels + [self.tokenizer.eos_token_id]
            if self.drop_meta and len(tokenized_text) > self.max_seq_length:
                return None
            # drop question to avoid no loss
            tokenized_text = tokenized_text[-self.max_seq_length:]
            labels = labels[-self.max_seq_length:]
            input_ids = torch.LongTensor(tokenized_text)
            labels = torch.LongTensor(labels)
            results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('base')
class BaseParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 split_paragraphs=False,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 prompt_type="empty",
                 content_key='text',
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        assert prompt_type in ["empty", "llama", "qwen", "intern", "baichuan2"], f"{prompt_type} has not supported."
        self.prompt_type = prompt_type
        self.drop_meta = drop_meta
        self.content_key = content_key
        self.split_paragraphs = split_paragraphs

    def build_input(self, raw_input_text):
        if self.prompt_type == "empty":
            prompt = raw_input_text
        elif self.prompt_type == "llama":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{raw_input_text}\n\n### Response:\n\n" # noqa
        elif self.prompt_type == "qwen":
            # pre_system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
            # prompt = f"\n<|im_start|>user\n{raw_input_text}<|im_end|>\n<|im_start|>assistant\n"
            prompt = f"user\n{raw_input_text}<|im_end|>\n<|im_start|>assistant\n"
        elif self.prompt_type == "intern":
            prompt = f"<|User|>:{raw_input_text}<eoh>\n<|Bot|>:"
        elif self.prompt_type == "baichuan2":
            prompt = f"{self.tokenizer.decode(195)}{raw_input_text}{self.tokenizer.decode(196)}"
        return prompt

    def __call__(self, meta):
        if self.content_key not in meta:
            raise ValueError(f"content_key {self.content_key} not in meta keys ({meta.keys()})")
        text = self.build_input(meta[self.content_key])
        if self.split_paragraphs:
            # split text into paragraphs and tokenize each paragraph separately
            # NOTE: The decoded text will have an additional whitespace at the beginning of each paragraph
            split_text = text.split('\n\n')
            tokenized_text = []
            for i in range(len(split_text)):
                add_special_tokens = (i == 0)  # only add special tokens for the first paragraph
                tokenized_text_tmp = self.tokenizer(
                    split_text[i] + '\n\n',
                    return_attention_mask=False,
                    add_special_tokens=add_special_tokens)['input_ids']
                tokenized_text += tokenized_text_tmp
        else:
            tokenized_text = self.tokenizer(
                text, return_attention_mask=False)['input_ids']
        if self.inference_mode:
            return tokenized_text, tokenized_text
        else:
            tokenized_text += [self.tokenizer.eos_token_id]
            if self.drop_meta and len(tokenized_text) > self.max_seq_length:
                return None
            # drop question to avoid no loss
            input_ids = torch.LongTensor(tokenized_text)[-self.max_seq_length:]
            labels = input_ids.clone()
            results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('general_chat')
class GeneralChatParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 only_last_answer=False,
                 prompt_template={},
                 inference_mode=False,
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.only_last_answer = only_last_answer
        self.system_prompt = prompt_template.get('system_prompt', "<系统> ")
        self.dialog_history_prompt = prompt_template.get('dialog_history_prompt', '<对话历史> ')
        self.question_prompt = prompt_template.get('question_prompt', "<最新问题> ")
        self.knowledge_prompt = prompt_template.get('knowledge_prompt', "<知识> ")
        self.answer_prompt = prompt_template.get('answer_prompt', "SenseChat:")
        self.user_prompt = prompt_template.get('user_prompt', "用户:")
        self.predefine_prompt = prompt_template.get('predefine_prompt', '')
        self.system_prompt = self.system_prompt + self.predefine_prompt
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta
        # <system>  <history>  <knowledge>  <question>  <answer>

    def convert2chat(self, meta):
        new_meta = []
        user, assistant = {}, {}
        if 'system' in meta:
            system = {}
            system['role'] = 'system'
            system['content'] = meta['system']
            new_meta.append(system)
        user['role'] = 'user'
        user['content'] = meta.get('instruction', '') + meta.get('input', "")
        assistant['role'] = 'assistant'
        assistant['content'] = meta.get('output', "")
        new_meta += [user, assistant]
        return new_meta

    def _process_meta(self, meta):
        '''
            process meta info:
            return
                system prompt
                knowledge_prompt
                history
                question =  =
                answer
        '''
        system_prompt_context = self.system_prompt
        if isinstance(meta, dict):
            if 'messages' in meta:
                if 'system' in meta:
                    system_prompt_context += meta['system']
                meta = meta['messages']
            if 'input' in meta:
                meta = self.convert2chat(meta)
        answer, question = "", ""
        dialog_history = []
        if len(meta) <= 1:
            for item in meta:
                if item['role'] == 'user':
                    question += item.get('content', '')
                if item['role'] == 'assistant':
                    answer += item.get('content', '')
        else:
            if self.inference_mode:
                question = meta[-1]['content']
                dialog_history = meta[:-1]
            else:
                answer = meta[-1]['content']
                question = meta[-2]['content']
                dialog_history = meta[:-2]

        knowledge_prompt_context = self.knowledge_prompt
        for item in dialog_history:
            if item['role'] == "system":
                system_prompt_context += item['content']
            if item['role'] == "knowledge":
                knowledge_prompt_context += item['content']
        rt_history = []
        for hist in dialog_history:
            if hist['role'] == "user" or hist['role'] == "assistant":
                rt_history.append(hist)
        return system_prompt_context, rt_history, knowledge_prompt_context, question, answer

    def _get_system_tokens_labels(self, system):
        tokens_system = self.tokenizer(system, return_attention_mask=False)['input_ids']
        labels_system = [self.ignore_index] * len(tokens_system)
        return tokens_system, labels_system

    def _get_history_tokens_labels(self, history):
        tokens_history = []
        labels_history = []
        for idx, item in enumerate(history):
            if item['role'] == "user":
                if idx == 0:
                    user_context = "{}{}{}".format(self.dialog_history_prompt, self.user_prompt, item['content'])
                else:
                    user_context = "{}{}".format(self.user_prompt, item['content'])
                token_user_context = self.tokenizer(user_context, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
                tokens_history += token_user_context
                labels_history += [self.ignore_index] * len(token_user_context)
            if item['role'] == "assistant":
                if idx == 0:
                    answer_prompt = "{}{}".format(self.dialog_history_prompt, self.answer_prompt)
                    # assis_context = "{}{}{}".format(self.dialog_history_prompt, self.answer_prompt, item['content'])
                else:
                    answer_prompt = self.answer_prompt
                tokens_answer_prompt = self.tokenizer(answer_prompt, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
                labels_answer_prompt = [self.ignore_index] * len(tokens_answer_prompt)
                tokens_answer = self.tokenizer(item['content'], return_attention_mask=False,
                                               add_special_tokens=False)['input_ids']
                if idx == 0:
                    tokens_answer_label = [self.ignore_index] * len(tokens_answer)
                else:
                    tokens_answer_label = copy.deepcopy(tokens_answer)
                tokens_answer += [self.tokenizer.eos_token_id]
                tokens_answer_label += [self.tokenizer.eos_token_id]
                token_assis_context = tokens_answer_prompt + tokens_answer
                label_assis_context = labels_answer_prompt + tokens_answer_label

                tokens_history += token_assis_context
                labels_history += label_assis_context
        return tokens_history, labels_history

    def _get_knowledge_tokens_labels(self, knowledge):
        tokens_knowledge = self.tokenizer(knowledge, return_attention_mask=False,
                                          add_special_tokens=False)['input_ids']
        labels_knowledge = [self.ignore_index] * len(tokens_knowledge)
        return tokens_knowledge, labels_knowledge

    def _get_question_tokens_labels(self, question):
        new_question = "{}{}{}".format(self.question_prompt, self.user_prompt, question)
        tokens_question = self.tokenizer(new_question, return_attention_mask=False,
                                         add_special_tokens=False)['input_ids']
        labels_question = [self.ignore_index] * len(tokens_question)
        return tokens_question, labels_question

    def _get_answer_tokens_labels(self, answer):
        tokens_answer_prompt = self.tokenizer(self.answer_prompt, return_attention_mask=False,
                                              add_special_tokens=False)['input_ids']
        labels_answer_prompt = [self.ignore_index] * len(tokens_answer_prompt)
        tokens_answer = self.tokenizer(answer, return_attention_mask=False,
                                       add_special_tokens=False)['input_ids']
        if not self.inference_mode:
            tokens_answer += [self.tokenizer.eos_token_id]
        labels_answer = labels_answer_prompt + tokens_answer
        tokens_answer = tokens_answer_prompt + tokens_answer
        return tokens_answer, labels_answer

    def build_inference_meta(self, text, history=[]):
        meta_dict = {}
        meta_dict['role'] = 'user'
        meta_dict['content'] = text
        meta = history + [meta_dict]
        return meta

    def get_tokens_labels(self, meta):
        system, history, knowledge, question, answer = self._process_meta(meta)

        tokens_system, labels_system = self._get_system_tokens_labels(system)
        tokens_history, labels_history = self._get_history_tokens_labels(history)
        tokens_knowledge, labels_knowledge = self._get_knowledge_tokens_labels(knowledge)
        tokens_question, labels_question = self._get_question_tokens_labels(question)
        tokens_answer, labels_answer = self._get_answer_tokens_labels(answer)

        # step1, clip history tokens from old to new
        tokens = tokens_system + tokens_history + tokens_knowledge + tokens_question + tokens_answer
        if len(tokens) > self.max_seq_length:
            if self.drop_meta:
                return None, None
            outside_length = len(tokens) - (self.max_seq_length + 1)

            tokens_history = tokens_history[outside_length:]
            labels_history = labels_history[outside_length:]

        tokens = tokens_system + tokens_history + tokens_knowledge + tokens_question + tokens_answer
        if self.only_last_answer:
            labels = [self.ignore_index] * len(labels_system + labels_history + labels_knowledge + labels_question) + labels_answer  # noqa
        else:
            labels = labels_system + labels_history + labels_knowledge + labels_question + labels_answer

        # step2, clip answer (When the history tokens is not enough to clip)
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        return tokens, labels

    def __call__(self, meta):
        if self.inference_mode:
            return self.get_tokens_labels(meta)
        tokens, labels = self.get_tokens_labels(meta)
        if tokens is None:
            return None
        input_ids = torch.LongTensor(tokens)
        if self.keep_all_keys:
            labels = input_ids.clone()
        else:
            labels = torch.LongTensor(labels)
        results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('combine_general_chat')
class CombinedGeneralChatParser(GeneralChatParser):
    """
    args are consistent with GeneralChatParser
    Combines the output of general_chat parsers in several conditions:
        - default_parser: parameters given in the yaml config file
        - last_answer_parser: only train last answer
        - keep_all_parser: keeps all keys containing answers and questions
    """
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 **kwargs):
        # self.default_parser = GeneralChatParser(**kwargs)

        super().__init__(tokenizer, max_seq_length, **kwargs)  # default parser is self

        updated_kwargs = dict(kwargs)
        updated_kwargs.update({'only_last_answer': True})
        self.last_answer_parser = GeneralChatParser(tokenizer, max_seq_length, **updated_kwargs)

        updated_kwargs.update({'only_last_answer': False, 'keep_all_keys': True})
        self.keep_all_parser = GeneralChatParser(tokenizer, max_seq_length, **updated_kwargs)

    def __call__(self, meta):
        # meta type 1: [convs]
        # meta type 2: {keep_all_keys: True, convs:[convs]}
        # meta type 3: {only_last_answer: True, convs:[convs]}
        if isinstance(meta, list):
            return super().__call__(meta)
        elif isinstance(meta, dict) and meta.get('keep_all_keys', False):
            return self.keep_all_parser(meta["convs"])
        elif isinstance(meta, dict) and meta.get('only_last_answer', False):
            return self.last_answer_parser(meta["convs"])
        else:
            return super().__call__(meta)


@PARSER_REGISTRY.register('combine_dpo_general_chat')
class CombinedDPOSFTParser(CombinedGeneralChatParser):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 inference_mode=False,
                 average_log_prob=False,
                 **kwargs):
        self.average_log_prob = average_log_prob
        assert inference_mode is False, 'dpo_rlhf parser does not support model inference'
        updated_kwargs = dict(kwargs)
        updated_kwargs.update({'inference_mode': inference_mode})
        super().__init__(tokenizer, max_seq_length, **kwargs)

    def parse_sft_pairs(self, meta):
        assert isinstance(meta['yw'], list) or isinstance(meta['yw'], dict)
        yw_res = super().__call__(meta['yw'])
        yl_res = super().__call__(meta['yl'])
        yw_tokens, yw_labels = yw_res['input_ids'], yw_res['labels']
        yl_tokens, yl_labels = yl_res['input_ids'], yl_res['labels']
        results = {'input_ids': [yw_tokens, yl_tokens], 'labels': [yw_labels, yl_labels], 'scores': torch.FloatTensor([999999, 999999])}
        return results

    def parse_dpo_pairs(self, meta):
        # DPO meta: {'yw':[convs], 'yl':[convs], 'yw_logp':[logp][1:], 'yl_logp':[logp][1:]}  only support one yw and one yl
        assert isinstance(meta['yw'], list) and isinstance(meta['yl'], list)

        yw_res = super().__call__({"convs": meta['yw'], "keep_all_keys": False, "only_last_answer": True})
        yl_res = super().__call__({"convs": meta['yl'], "keep_all_keys": False, "only_last_answer": True})
        yw_tokens, yw_labels = yw_res['input_ids'], yw_res['labels']
        yl_tokens, yl_labels = yl_res['input_ids'], yl_res['labels']
        yw_logp = torch.FloatTensor(meta['yw_logp'])
        yl_logp = torch.FloatTensor(meta['yl_logp'])
        yw_mask = yw_labels != self.ignore_index
        yl_mask = yl_labels != self.ignore_index

        if self.average_log_prob:
            yw_score = (yw_logp * yw_mask[1:]).sum(-1) / yw_mask[1:].sum(-1)  # LLMs output the probobilities from the second token
            yl_score = (yl_logp * yl_mask[1:]).sum(-1) / yl_mask[1:].sum(-1)
        else:
            yw_score = (yw_logp * yw_mask[1:]).sum(-1)
            yl_score = (yl_logp * yl_mask[1:]).sum(-1)

        results = {'input_ids': [yw_tokens, yl_tokens], 'labels': [yw_labels, yl_labels], 'scores': torch.FloatTensor([yw_score, yl_score])}
        return results

    def __call__(self, meta):
        yw_logp = torch.FloatTensor(meta['yw_logp'])
        if len(yw_logp) == 1 and yw_logp[0] >= 1:  # log prob>0 means it's sft data
            return self.parse_sft_pairs(meta)
        else:
            return self.parse_dpo_pairs(meta)


@PARSER_REGISTRY.register('simple_chat')
class SimpleChatParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 only_last_answer=False,
                 prompt_template={},
                 inference_mode=False,
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.only_last_answer = only_last_answer
        self.system_prompt = prompt_template.get('system_prompt', "<|System>|>:")
        self.question_prompt = prompt_template.get('qustion_prompt', "<|Human|>:")
        self.answer_prompt = prompt_template.get('answer_prompt', "<|Assistant|>:")
        self.predefine_prompt = prompt_template.get('predefine_prompt', '')
        self.system_prompt = self.system_prompt + self.predefine_prompt
        self.inference_mode = inference_mode
        self.eoh = prompt_template.get('eoh', "\n")
        self.eosys = prompt_template.get('eosys', "\n")
        self.eoa = prompt_template.get('eoa', '')
        self.drop_meta = drop_meta

    def convert2chat(self, meta):
        new_meta = []
        user, assistant = {}, {}
        if 'system' in meta:
            system = {}
            system['role'] = 'system'
            system['content'] = meta['system']
            new_meta.append(system)
        user['role'] = 'user'
        user['content'] = meta.get('instruction', '') + meta.get('input', "")
        assistant['role'] = 'assistant'
        assistant['content'] = meta.get('output', "")
        new_meta += [user, assistant]
        return new_meta

    def _process_meta(self, meta):
        '''
            process meta info:
            return
                system prompt
                history
                question
                answer
        '''
        system_prompt_context = self.system_prompt
        if isinstance(meta, dict):
            if 'messages' in meta:
                if 'system' in meta:
                    system_prompt_context += meta['system']
                meta = meta['messages']
            if 'input' in meta:
                meta = self.convert2chat(meta)
        answer, question = "", ""
        dialog_history = []
        if len(meta) <= 1:
            for item in meta:
                if item['role'] == 'user':
                    question += item.get('content', '')
                if item['role'] == 'assistant':
                    answer += item.get('content', '')
        else:
            if self.inference_mode:
                question = meta[-1]['content']
                dialog_history = meta[:-1]
            else:
                answer = meta[-1]['content']
                question = meta[-2]['content']
                dialog_history = meta[:-2]
        for item in dialog_history:
            if item['role'] == "system":
                system_prompt_context += item['content']
        rt_history = []
        for hist in dialog_history:
            if hist['role'] == "user" or hist['role'] == "assistant":
                rt_history.append(hist)
        system_prompt_context = "{}{}".format(system_prompt_context, self.eosys)
        return system_prompt_context, rt_history, question, answer

    def _get_system_tokens_labels(self, system):
        tokens_system = self.tokenizer(system, return_attention_mask=False)['input_ids']
        labels_system = [self.ignore_index] * len(tokens_system)
        return tokens_system, labels_system

    def _get_history_tokens_labels(self, history):
        tokens_history = []
        labels_history = [self.ignore_index] * len(tokens_history)
        for idx, item in enumerate(history):
            if item['role'] == "user":
                user_context = "{}{}{}".format(self.question_prompt, item['content'], self.eoh)
                token_user_context = self.tokenizer(user_context, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
                tokens_history += token_user_context
                labels_history += [self.ignore_index] * len(token_user_context)
            if item['role'] == "assistant":
                token_answer_prompt = self.tokenizer(self.answer_prompt, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
                labels_answer_prompt = [self.ignore_index] * len(token_answer_prompt)
                assis_context = "{}{}".format(item['content'], self.eoa)
                token_assis_context = self.tokenizer(assis_context, return_attention_mask=False, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]  # noqa
                # final_label
                labels_answer_prompt = labels_answer_prompt + token_assis_context
                # final_tokens
                token_assis_context = token_answer_prompt + token_assis_context
                if idx == 0:
                    labels_answer_prompt = [self.ignore_index] * len(labels_answer_prompt)
                tokens_history += token_assis_context
                labels_history += labels_answer_prompt
        return tokens_history, labels_history

    def _get_question_tokens_labels(self, question):
        new_question = "{}{}{}".format(self.question_prompt, question, self.eoh)
        tokens_question = self.tokenizer(new_question, return_attention_mask=False,
                                         add_special_tokens=False)['input_ids']
        labels_question = [self.ignore_index] * len(tokens_question)
        return tokens_question, labels_question

    def _get_answer_tokens_labels(self, answer):
        answer_prompt_tokens = self.tokenizer(self.answer_prompt, return_attention_mask=False,
                                              add_special_tokens=False)['input_ids']
        labels_answer_prompt = [self.ignore_index] * len(answer_prompt_tokens)
        if not self.inference_mode:
            answer += self.eoa
        tokens_answer = self.tokenizer(answer, return_attention_mask=False,
                                       add_special_tokens=False)['input_ids']
        if not self.inference_mode:
            tokens_answer += [self.tokenizer.eos_token_id]
        labels_answer = labels_answer_prompt + tokens_answer
        tokens_answer = answer_prompt_tokens + tokens_answer
        return tokens_answer, labels_answer

    def build_inference_meta(self, text, history=[]):
        meta_dict = {}
        meta_dict['role'] = 'user'
        meta_dict['content'] = text
        meta = history + [meta_dict]
        return meta

    def get_tokens_labels(self, meta):
        system, history, question, answer = self._process_meta(meta)
        tokens_system, labels_system = self._get_system_tokens_labels(system)
        tokens_history, labels_history = self._get_history_tokens_labels(history)
        tokens_question, labels_question = self._get_question_tokens_labels(question)
        tokens_answer, labels_answer = self._get_answer_tokens_labels(answer)

        tokens = tokens_system + tokens_history + tokens_question + tokens_answer
        if self.only_last_answer:
            labels = [self.ignore_index] * len(labels_system + labels_history + labels_question) + labels_answer  # noqa
        else:
            labels = labels_system + labels_history + labels_question + labels_answer

        if len(tokens) > self.max_seq_length:
            if self.drop_meta:
                return None, None
            outside_length = len(tokens) - self.max_seq_length
            # step1, clip history tokens from old to new
            tokens_history = tokens_history[outside_length:]
            labels_history = labels_history[outside_length:]
            tokens = tokens_system + tokens_history + tokens_question + tokens_answer
            labels = labels_system + labels_history + labels_question + labels_answer
            # step2, clip answer (When the history tokens is not enough to clip)
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
                labels = labels[:self.max_seq_length]
        return tokens, labels

    def __call__(self, meta):
        if self.inference_mode:
            return self.get_tokens_labels(meta)
        tokens, labels = self.get_tokens_labels(meta)
        if tokens is None:
            return None
        input_ids = torch.LongTensor(tokens)
        if self.keep_all_keys:
            labels = input_ids.clone()
        else:
            labels = torch.LongTensor(labels)
        results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('reward')
class RewardParser(BaseParser):
    def __init__(self, tokenizer, max_seq_length, ignore_index=-100,
                 keep_all_keys=False, return_type='token_ids', inference_mode=False):
        super().__init__(tokenizer, max_seq_length, ignore_index, keep_all_keys, return_type)
        self.inference_mode = inference_mode

    def __call__(self, meta):
        input_text = meta['input_text']
        choice = meta['choice']
        bad_answer = meta.get('bad_answer', '')
        input_text_token = self.tokenizer(input_text, return_attention_mask=False)['input_ids']
        choice_token = self.tokenizer(choice, return_attention_mask=False, add_special_tokens=False)['input_ids']
        if not self.inference_mode:
            bad_answer_token = self.tokenizer(bad_answer, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
            choice_input = input_text_token + choice_token + [self.tokenizer.eos]
            bad_input = input_text_token + bad_answer_token + [self.tokenizer.eos]
            choice_input_ids = torch.LongTensor(choice_input)
            bad_input_ids = torch.LongTensor(bad_input)
            if len(choice_input_ids) > self.max_seq_length:
                choice_input_ids = choice_input_ids[:self.max_seq_length]
            if len(bad_input_ids) > self.max_seq_length:
                bad_input_ids = bad_input_ids[:self.max_seq_length]
            results = {'input_ids': (choice_input_ids, bad_input_ids)}
        else:
            length = len(input_text_token) + len(choice_token)
            if length > self.max_seq_length:
                input_text_token = input_text_token[length - self.max_seq_length:]
            labels = torch.LongTensor([self.ignore_index] * len(input_text_token) + choice_token)[:self.max_seq_length]
            input_ids = input_text_token + choice_token
            input_ids = torch.LongTensor(input_ids)
            results = {'input_ids': input_ids, "labels": labels}
        return results


@PARSER_REGISTRY.register('mini_rlhf')
class MiniRLHFParser(BaseParser):
    def __init__(self, tokenizer, max_seq_length, ignore_index=-100,
                 keep_all_keys=False, return_type='token_ids', inference_mode=False):
        super().__init__(tokenizer, max_seq_length, ignore_index, keep_all_keys, return_type)
        self.inference_mode = inference_mode

    def __call__(self, meta):
        question = meta['question']
        answers = meta['answers']
        scores = meta['scores']
        input_question_token = self.tokenizer(question, return_attention_mask=False)['input_ids']
        if not self.inference_mode:
            assert len(self.tokenizer.eos_token), 'the tokenizer must have an eos_token, please check your special_tokens_map.json or set it manually'      # noqa
            all_sentense_token = []
            all_labels = []
            for ans in answers:
                ans = f"{ans}{self.tokenizer.eos_token}"
                ans_token = self.tokenizer(ans, return_attention_mask=False,
                                           add_special_tokens=False)['input_ids']
                sentense_token = torch.LongTensor(input_question_token + ans_token)[-self.max_seq_length:]
                all_sentense_token.append(sentense_token)     # noqa
                labels = torch.LongTensor([self.ignore_index] * len(input_question_token) + ans_token)[-self.max_seq_length:]        # noqa
                all_labels.append(labels)
            results = {'input_ids': all_sentense_token, 'labels': all_labels, 'scores': torch.FloatTensor(scores)}
        else:
            results = question
        return results


@PARSER_REGISTRY.register('QWen_Vl')
class QWenVLParser(SimpleChatParser):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 prompt_template={},
                 inference_mode=False,
                 only_last_answer=False,
                 keep_all_keys=False,
                 drop_meta=False):
        super().__init__(tokenizer, max_seq_length, ignore_index, keep_all_keys, only_last_answer, prompt_template, inference_mode, drop_meta)
        self.system_prompt = prompt_template.get('system_prompt', "You are a helpful assistant.")
        self.nl_tokens = self.tokenizer('\n').input_ids
        self.im_start = self.tokenizer.im_start_id
        self.im_end = self.tokenizer.im_end_id

    def _get_system_tokens_labels(self, system):
        system = "<|im_start|>system\n" + system
        tokens_system = self.tokenizer(system, return_attention_mask=False)['input_ids']
        labels_system = [self.im_start] + [self.ignore_index] * (len(tokens_system) - 3) + [self.im_end] + self.nl_tokens
        return tokens_system, labels_system

    def _get_question_tokens_labels(self, question):
        new_question = "{}{}{}".format(self.question_prompt, question, self.eoh)
        tokens_question = self.tokenizer(new_question, return_attention_mask=False,
                                         add_special_tokens=False)['input_ids']
        labels_question = [self.im_start] + [self.ignore_index] * (len(tokens_question) - 3) + [self.im_end] + self.nl_tokens
        return tokens_question, labels_question

    def _get_history_tokens_labels(self, history):
        tokens_history = []
        labels_history = []
        for idx, item in enumerate(history):
            if item['role'] == "user":
                token_user_context = self.tokenizer("<|im_start|>user").input_ids + self.nl_tokens + \
                    self.tokenizer(item['content']).input_ids + [self.im_end] + self.nl_tokens
                labels_user_context = [self.im_start] + [self.ignore_index] * (len(token_user_context) - 3) + [self.im_end] + self.nl_tokens
                tokens_history += token_user_context
                labels_history += labels_user_context
            if item['role'] == "assistant":
                token_assis_context = self.tokenizer("<|im_start|>assistant").input_ids + self.nl_tokens + \
                    self.tokenizer(item['content']).input_ids + [self.im_end] + self.nl_tokens
                labels_answer_prompt = [self.im_start] + [self.ignore_index] * len(self.tokenizer("<|im_start|>assistant").input_ids) + \
                    token_assis_context[len(self.tokenizer("<|im_start|>assistant").input_ids) + 1:-2] + [self.im_end] + self.nl_tokens
                if idx == 0:
                    labels_answer_prompt = [self.ignore_index] * len(labels_answer_prompt)
                tokens_history += token_assis_context
                labels_history += labels_answer_prompt
        return tokens_history, labels_history

    def _get_answer_tokens_labels(self, answer):
        answer_prompt_tokens = self.tokenizer(self.answer_prompt, return_attention_mask=False, add_special_tokens=False)['input_ids']
        labels_answer_prompt = [self.im_start] + [self.ignore_index] * (len(answer_prompt_tokens))
        if not self.inference_mode:
            answer += self.eoa
        tokens_answer = self.tokenizer(answer, return_attention_mask=False, add_special_tokens=False)['input_ids']
        if not self.inference_mode:
            tokens_answer = tokens_answer + [self.im_end] + self.nl_tokens
        labels_answer = labels_answer_prompt + tokens_answer
        tokens_answer = answer_prompt_tokens + tokens_answer
        labels_answer = [self.im_start] + [self.ignore_index] * len(self.tokenizer("<|im_start|>assistant").input_ids) + \
            tokens_answer[len(self.tokenizer("<|im_start|>assistant").input_ids) + 1:-2] + [self.im_end] + self.nl_tokens
        return tokens_answer, labels_answer


@AUGMENTATION_REGISTRY.register('sense_tokenization')
class SenseTokenization(object):
    def __init__(self, tokenizer, max_seq_length, parser_type=None, parser_kwargs={}, ignore_index=-100):
        parser_kwargs.update({'tokenizer': tokenizer, 'max_seq_length': max_seq_length,
                              'ignore_index': ignore_index})
        assert ('default' not in PARSER_REGISTRY) and ('chat' not in PARSER_REGISTRY), 'defualt and chat are keeped for sense tokenization, you can not register them.'     # noqa
        if parser_type is None or parser_type == 'default':
            parser_type = 'sense'
        elif parser_type == 'chat':
            parser_type = 'sense_chat'
        self.parser = build_parser(parser_type, parser_kwargs)

    def __call__(self, *args, **kwargs):
        return self.parser(*args, **kwargs)


def build_parser(parser_type, parser_kwargs):
    return PARSER_REGISTRY.build({"type": parser_type, "kwargs": parser_kwargs})


def build_augmentation(cfg):
    if 'template' in cfg['kwargs']:
        cfg['kwargs'].pop('template')
    return AUGMENTATION_REGISTRY.build(cfg)


def build_transformer(cfgs):
    transform_list = [build_augmentation(cfg) for cfg in cfgs]
    return NLPCompose(transform_list)
