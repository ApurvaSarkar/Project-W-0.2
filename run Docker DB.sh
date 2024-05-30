docker run -p 5532:5432 -d -e POSTGRES_DB=ai -e POSTGRES_USER=ai -e POSTGRES_PASSWORD=ai -e PGDATA=/var/lib/postgresql/data/pgdata -v pgvolume:/var/lib/postgresql/data  --name pgvector phidata/pgvector:16

docker exec -it 24bf5356c5aa psql -U ai

\list

\dt

SELECT * FROM local_rag_assistant;