#!/bin/bash
# Quick helper script for visualizing GPSD policies

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   GPSD Policy Visualization Helper    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: python3 not found${NC}"
    exit 1
fi

# Function to list policies
list_policies() {
    echo -e "${GREEN}Available trained policies:${NC}"
    python3 test_gpsd.py --policy list
}

# Function to test latest policy
test_latest() {
    LATEST=$(ls -t runs/*/gpsd_ppo_agent.pt 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo -e "${YELLOW}No trained policies found. Train a model first!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Testing latest policy: $LATEST${NC}"
    python3 test_gpsd.py --policy "$LATEST" "$@"
}

# Function to compare all policies
compare_all() {
    echo -e "${GREEN}Comparing all trained policies...${NC}"
    python3 compare_policies.py --policies all --num-episodes 10 --include-random "$@"
}

# Function to compare recent policies
compare_recent() {
    N=${1:-3}
    echo -e "${GREEN}Comparing $N most recent policies...${NC}"
    POLICIES=$(ls -t runs/*/gpsd_ppo_agent.pt 2>/dev/null | head -$N | tr '\n' ' ')
    if [ -z "$POLICIES" ]; then
        echo -e "${YELLOW}No trained policies found. Train a model first!${NC}"
        exit 1
    fi
    shift
    python3 compare_policies.py --policies $POLICIES --num-episodes 10 --include-random "$@"
}

# Main menu
case "$1" in
    list|ls)
        list_policies
        ;;
    latest)
        shift
        test_latest "$@"
        ;;
    compare)
        shift
        compare_all "$@"
        ;;
    compare-recent)
        shift
        compare_recent "$@"
        ;;
    test)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Usage: $0 test <policy_path> [options]${NC}"
            exit 1
        fi
        shift
        python3 test_gpsd.py --policy "$@"
        ;;
    random)
        shift
        echo -e "${GREEN}Testing random policy (baseline)${NC}"
        python3 test_gpsd.py --policy random "$@"
        ;;
    help|--help|-h)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  list              - List all available trained policies"
        echo "  latest            - Test the most recently trained policy with visualization"
        echo "  random            - Test random baseline policy"
        echo "  test <path>       - Test a specific policy (provide path to .pt file)"
        echo "  compare           - Compare all trained policies (statistical analysis)"
        echo "  compare-recent [N]- Compare N most recent policies (default: 3)"
        echo "  help              - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 list"
        echo "  $0 latest"
        echo "  $0 latest --num-episodes 3"
        echo "  $0 random"
        echo "  $0 test runs/gpsd__train_gpsd_ppo__1__1770837747/gpsd_ppo_agent.pt"
        echo "  $0 compare"
        echo "  $0 compare --save-plot results.png"
        echo "  $0 compare-recent 5"
        echo ""
        echo "Additional options can be passed to the underlying scripts."
        echo "See VISUALIZATION_README.md for detailed documentation."
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
