from __future__ import annotations
import json
import logging
from typing import Literal
import ftfy
import numpy as np
from pydantic import BaseModel
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatMessage
from autogpt.processing.text import chunk_content, split_text, summarize_text
from .utils import Embedding, get_embedding
logger = logging.getLogger(__name__)
MemoryDocType = Literal['webpage', 'text_file', 'code_file', 'agent_history']

class MemoryItem(BaseModel, arbitrary_types_allowed=True):
    """Memory object containing raw content as well as embeddings"""
    raw_content: str
    summary: str
    chunks: list[str]
    chunk_summaries: list[str]
    e_summary: Embedding
    e_chunks: list[Embedding]
    metadata: dict

    def relevance_for(self, query: str, e_query: Embedding | None=None):
        if False:
            for i in range(10):
                print('nop')
        return MemoryItemRelevance.of(self, query, e_query)

    @staticmethod
    def from_text(text: str, source_type: MemoryDocType, config: Config, metadata: dict={}, how_to_summarize: str | None=None, question_for_summary: str | None=None):
        if False:
            i = 10
            return i + 15
        logger.debug(f"Memorizing text:\n{'-' * 32}\n{text}\n{'-' * 32}\n")
        text = ftfy.fix_text(text)
        chunks = [chunk for (chunk, _) in (split_text(text, config.embedding_model, config) if source_type != 'code_file' else chunk_content(text, config.embedding_model))]
        logger.debug('Chunks: ' + str(chunks))
        chunk_summaries = [summary for (summary, _) in [summarize_text(text_chunk, config, instruction=how_to_summarize, question=question_for_summary) for text_chunk in chunks]]
        logger.debug('Chunk summaries: ' + str(chunk_summaries))
        e_chunks = get_embedding(chunks, config)
        summary = chunk_summaries[0] if len(chunks) == 1 else summarize_text('\n\n'.join(chunk_summaries), config, instruction=how_to_summarize, question=question_for_summary)[0]
        logger.debug('Total summary: ' + summary)
        e_summary = get_embedding(summary, config)
        metadata['source_type'] = source_type
        return MemoryItem(raw_content=text, summary=summary, chunks=chunks, chunk_summaries=chunk_summaries, e_summary=e_summary, e_chunks=e_chunks, metadata=metadata)

    @staticmethod
    def from_text_file(content: str, path: str, config: Config):
        if False:
            i = 10
            return i + 15
        return MemoryItem.from_text(content, 'text_file', config, {'location': path})

    @staticmethod
    def from_code_file(content: str, path: str):
        if False:
            for i in range(10):
                print('nop')
        return MemoryItem.from_text(content, 'code_file', {'location': path})

    @staticmethod
    def from_ai_action(ai_message: ChatMessage, result_message: ChatMessage):
        if False:
            i = 10
            return i + 15
        if ai_message.role != 'assistant':
            raise ValueError(f"Invalid role on 'ai_message': {ai_message.role}")
        result = result_message.content if result_message.content.startswith('Command') else 'None'
        user_input = result_message.content if result_message.content.startswith('Human feedback') else 'None'
        memory_content = f'Assistant Reply: {ai_message.content}\n\nResult: {result}\n\nHuman Feedback: {user_input}'
        return MemoryItem.from_text(text=memory_content, source_type='agent_history', how_to_summarize="if possible, also make clear the link between the command in the assistant's response and the command result. Do not mention the human feedback if there is none")

    @staticmethod
    def from_webpage(content: str, url: str, config: Config, question: str | None=None):
        if False:
            i = 10
            return i + 15
        return MemoryItem.from_text(text=content, source_type='webpage', config=config, metadata={'location': url}, question_for_summary=question)

    def dump(self, calculate_length=False) -> str:
        if False:
            return 10
        if calculate_length:
            token_length = self.llm_provider.count_tokens(self.raw_content, Config().embedding_model)
        return f"\n=============== MemoryItem ===============\nSize: {(f'{token_length} tokens in ' if calculate_length else '')}{len(self.e_chunks)} chunks\nMetadata: {json.dumps(self.metadata, indent=2)}\n---------------- SUMMARY -----------------\n{self.summary}\n------------------ RAW -------------------\n{self.raw_content}\n==========================================\n"

    def __eq__(self, other: MemoryItem):
        if False:
            i = 10
            return i + 15
        return self.raw_content == other.raw_content and self.chunks == other.chunks and (self.chunk_summaries == other.chunk_summaries) and np.array_equal(self.e_summary if isinstance(self.e_summary, np.ndarray) else np.array(self.e_summary, dtype=np.float32), other.e_summary if isinstance(other.e_summary, np.ndarray) else np.array(other.e_summary, dtype=np.float32)) and np.array_equal(self.e_chunks if isinstance(self.e_chunks[0], np.ndarray) else [np.array(c, dtype=np.float32) for c in self.e_chunks], other.e_chunks if isinstance(other.e_chunks[0], np.ndarray) else [np.array(c, dtype=np.float32) for c in other.e_chunks])

class MemoryItemRelevance(BaseModel):
    """
    Class that encapsulates memory relevance search functionality and data.
    Instances contain a MemoryItem and its relevance scores for a given query.
    """
    memory_item: MemoryItem
    for_query: str
    summary_relevance_score: float
    chunk_relevance_scores: list[float]

    @staticmethod
    def of(memory_item: MemoryItem, for_query: str, e_query: Embedding | None=None) -> MemoryItemRelevance:
        if False:
            for i in range(10):
                print('nop')
        e_query = e_query if e_query is not None else get_embedding(for_query)
        (_, srs, crs) = MemoryItemRelevance.calculate_scores(memory_item, e_query)
        return MemoryItemRelevance(for_query=for_query, memory_item=memory_item, summary_relevance_score=srs, chunk_relevance_scores=crs)

    @staticmethod
    def calculate_scores(memory: MemoryItem, compare_to: Embedding) -> tuple[float, float, list[float]]:
        if False:
            i = 10
            return i + 15
        '\n        Calculates similarity between given embedding and all embeddings of the memory\n\n        Returns:\n            float: the aggregate (max) relevance score of the memory\n            float: the relevance score of the memory summary\n            list: the relevance scores of the memory chunks\n        '
        summary_relevance_score = np.dot(memory.e_summary, compare_to)
        chunk_relevance_scores = np.dot(memory.e_chunks, compare_to).tolist()
        logger.debug(f'Relevance of summary: {summary_relevance_score}')
        logger.debug(f'Relevance of chunks: {chunk_relevance_scores}')
        relevance_scores = [summary_relevance_score, *chunk_relevance_scores]
        logger.debug(f'Relevance scores: {relevance_scores}')
        return (max(relevance_scores), summary_relevance_score, chunk_relevance_scores)

    @property
    def score(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'The aggregate relevance score of the memory item for the given query'
        return max([self.summary_relevance_score, *self.chunk_relevance_scores])

    @property
    def most_relevant_chunk(self) -> tuple[str, float]:
        if False:
            i = 10
            return i + 15
        'The most relevant chunk of the memory item + its score for the given query'
        i_relmax = np.argmax(self.chunk_relevance_scores)
        return (self.memory_item.chunks[i_relmax], self.chunk_relevance_scores[i_relmax])

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.memory_item.summary} ({self.summary_relevance_score}) {self.chunk_relevance_scores}'