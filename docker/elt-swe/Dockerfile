# Agent execution image for official ELT-Bench tasks.
# Same tool set as ELT-Bench's agents/SWE-agent/docker/elt-swe, pinned to Debian bookworm:
# the upstream Dockerfile installs software-properties-common, which no longer exists on
# trixie (python:3.11's current base), so the upstream build fails as of 2026-10.
FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    iputils-ping gnupg wget lsb-release curl jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir \
    pyarrow db-dtypes pandas numpy openpyxl pyyaml requests \
    snowflake-connector-python dbt-core dbt-snowflake dbt-databricks dbt-redshift \
    databricks-sdk databricks-sql-connector redshift-connector \
    psycopg2-binary pymongo flask awscli

RUN wget -qO- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/hashicorp.list \
    && apt-get update && apt-get install -y --no-install-recommends terraform && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["/bin/bash"]
