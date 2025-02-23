"""create tables figi

Revision ID: 9427eb6790f9
Revises: 7ddda8fd22bc
Create Date: 2025-02-21 14:52:01.825914

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9427eb6790f9'
down_revision: Union[str, None] = '7ddda8fd22bc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    sql = """
    CREATE TABLE figi (
    id SERIAL PRIMARY KEY,
    figi VARCHAR(20) NOT NULL,
    isin VARCHAR(20) NOT NULL,
    class_code VARCHAR(20) NOT NULL,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(255) NOT NULL,
    instrument_type VARCHAR(255) NOT NULL,
    UNIQUE (figi, ticker) -- Обязательно для upsert
    )    
    """
    op.execute(sql)
    #op.create_table(
    #    'quotes',
    #    sa.Column('id', sa.Integer, primary_key=True),
    #    sa.Column('figi', sa.String(20), nullable=False),
    #    sa.Column('description', sa.Unicode(200)),
    #)

def downgrade() -> None:
    op.drop_table('figi')


