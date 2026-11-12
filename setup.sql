-- Enable the vector extension
create extension if not exists vector
with schema extensions;

-- Create the documents table
create table documents (
  id bigint primary key generated always as identity,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding extensions.vector(384)
);

-- Create an index for faster similarity search
create index on documents
using hnsw (embedding extensions.vector_cosine_ops);

-- Create the match_documents function
create or replace function match_documents (
  query_embedding extensions.vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by (documents.embedding <=> query_embedding) asc
  limit match_count;
$$;