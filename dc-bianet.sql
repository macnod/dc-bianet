    drop table environments cascade;
    drop table weight_sets cascade;
	drop table neurons cascade;
	drop table connections cascade;
	drop table weights;
	drop table output_labels;
	drop table training cascade;
	drop table error;
	drop table sample_partitions cascade;
	drop table samples cascade;
	drop table vectors;

    create table environments (
        id serial primary key,
        environment_name text not null,
        learning_rate real not null,
        momentum real not null,
        weight_reset_function_name text not null,
        log_file text not null
    );

    create table neurons (
        id serial primary key,
        environment_id integer references environments,
        bianet_id text not null,
        neuron_name text not null,
        neuron_input real not null,
        neuron_output real not null,
        layer integer not null,
        biased boolean not null,
        transfer_key text not null
    );

    create table connections (
        id serial primary key,
        environment_id integer references environments,
        source_id integer references neurons,
        target_id integer references neurons,
        learning_rate real not null,
        momentum real not null,
        delta real not null
    );

    create table weight_sets (
        id serial primary key,
        environment_id integer references environments,
        weight_set_name text not null
    );

    create table weights (
        id serial primary key,
        environment_id integer references environments,
        weight_set_id integer references weight_sets
    );
        
    create table output_labels (
        id serial primary key,
        environment_id integer references environments,
        output_number integer not null,
        label text
    );

    create table sample_partitions (
        id serial primary key,
        environment_id integer references environments,
        partition_name text not null
    );

    create table samples (
        id serial primary key,
        environment_id integer references environments,
        partition_id integer references sample_partitions
    );

    create table training (
        id serial primary key,
        environment_id integer references environments,
        max_epochs integer not null,
        target_error real not null,
        thread_count integer not null,
        report_frequency integer not null,
        weight_set_id integer references weight_sets,
        sample_id integer references samples
    );

    create table error (
        id serial primary key,
        environment_id integer references environments,
        training_id integer references training,
        time_of_entry timestamp not null,
        elapsed_seconds integer not null,
        iteration integer not null,
        iteration_vector_count integer not null,
        total_vector_count integer not null,
        vectors_per_second real not null,
        error_value real not null
    );

    create table vectors (
        id serial primary key,
        environment_id integer references environments,
        sample_id integer references samples(id),
        inputs real array,
        outputs real array
    );
