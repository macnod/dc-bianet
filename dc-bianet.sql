-- drop table if exists samples cascade;
-- create table samples (
--     id serial primary key,
--     environment text not null,
--     partition text not null
-- );

-- drop table if exists vector;
-- create table vector (
--     id serial primary key,
--     sample_id integer references samples(id),
--     inputs real array,
--     outputs real array
-- );

-- drop index if exists samples_environment_idx;
-- create index samples_environment_idx on samples(environment);
-- drop index if exists vector_sample_id_idx;
-- create index vector_sample_id_idx on vector(sample_id);


create table training_sets (
    id serial primary key,
    name text not null
);


    
    
